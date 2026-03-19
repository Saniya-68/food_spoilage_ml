from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, abort
import sqlite3
import os
import logging
from datetime import datetime, timedelta
import joblib
from utils.recipe_engine import RecipeEngine
from utils.email_utils import NotificationManager
from utils.ml_utils import predict_days

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "supersecretkey")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "database.db")
MODEL_PATH = os.path.join(BASE_DIR, "model", "food_spoilage_pipeline.pkl")

EMAIL_USER = os.getenv("EMAIL_USER", "projfoodtracker@gmail.com")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "kqqjcfhbrsdhzoas")

app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = EMAIL_USER
app.config["MAIL_PASSWORD"] = EMAIL_PASSWORD

logging.basicConfig(
    filename=os.path.join(BASE_DIR, "app.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run python train_model.py first.")

model_pipeline = joblib.load(MODEL_PATH)
recipe_engine = RecipeEngine()
notification_manager = NotificationManager(None, EMAIL_USER)


def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    c = conn.cursor()

    c.execute(
        """CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )"""
    )

    # Base items table definition
    c.execute(
        """CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            food_type TEXT NOT NULL,
            storage_type TEXT NOT NULL,
            purchase_date TEXT NOT NULL,
            temperature REAL NOT NULL,
            humidity REAL NOT NULL,
            predicted_days INTEGER NOT NULL,
            expiry_date TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )"""
    )

    # Add missing columns from previous schema versions
    existing_columns = [r[1] for r in c.execute("PRAGMA table_info(items)").fetchall()]
    needed_columns = {
        "food_type": "TEXT NOT NULL DEFAULT 'Unknown'",
        "storage_type": "TEXT NOT NULL DEFAULT 'Room'",
        "purchase_date": "TEXT NOT NULL DEFAULT '2000-01-01'",
        "temperature": "REAL NOT NULL DEFAULT 20.0",
        "humidity": "REAL NOT NULL DEFAULT 50.0",
        "predicted_days": "INTEGER NOT NULL DEFAULT 0",
        "created_at": "TEXT NOT NULL DEFAULT '2000-01-01T00:00:00'",
    }
    for col, ddl in needed_columns.items():
        if col not in existing_columns:
            c.execute(f"ALTER TABLE items ADD COLUMN {col} {ddl}")

    conn.commit()
    conn.close()


def validate_date(date_text):
    try:
        return datetime.strptime(date_text, "%Y-%m-%d").date()
    except Exception:
        return None


def classify_status(days_left):
    if days_left < 0:
        return "Expired"
    if days_left <= 2:
        return "Expiring Soon"
    return "Fresh"


def get_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    conn = get_db_connection()
    user = conn.execute("SELECT id, username, email FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return user


def check_and_send_notifications(user):
    conn = get_db_connection()
    rows = conn.execute("SELECT name, expiry_date, status FROM items WHERE user_id = ?", (user["id"],)).fetchall()
    conn.close()
    for item in rows:
        expiry_date = validate_date(item["expiry_date"])
        if not expiry_date:
            continue
        days_left = (expiry_date - datetime.now().date()).days
        if days_left <= 2 and item["status"] != "Expired":
            logging.info(f"Expiry alert for user {user['username']} item {item['name']} days_left={days_left}")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        if not (username and email and password):
            flash("Please fill in all fields", "danger")
            return render_template("register.html")
        conn = get_db_connection()
        try:
            conn.execute(
                "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                (username, email, password),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            flash("Username or email already exists", "warning")
            conn.close()
            return render_template("register.html")
        conn.close()
        flash("Registration successful. Please login.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        conn = get_db_connection()
        user = conn.execute(
            "SELECT id, username, email FROM users WHERE username = ? AND password = ?",
            (username, password),
        ).fetchone()
        conn.close()
        if user:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash("Logged in successfully", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid username/password", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def dashboard():
    user = get_user()
    if not user:
        return redirect(url_for("login"))
    filter_status = request.args.get("status", "all")
    conn = get_db_connection()
    query = "SELECT * FROM items WHERE user_id = ?"
    params = [user["id"]]
    if filter_status in ["Fresh", "Expiring Soon", "Expired"]:
        query += " AND status = ?"
        params.append(filter_status)
    rows = conn.execute(query, params).fetchall()
    conn.close()

    items = [dict(item) for item in rows]
    analytics = {
        "total": len(items),
        "fresh": sum(1 for i in items if i["status"] == "Fresh"),
        "expiring": sum(1 for i in items if i["status"] == "Expiring Soon"),
        "expired": sum(1 for i in items if i["status"] == "Expired"),
    }
    check_and_send_notifications(user)
    return render_template("dashboard.html", items=items, analytics=analytics, username=user["username"])


@app.route("/add", methods=["GET", "POST"])
def add_item():
    user = get_user()
    if not user:
        return redirect(url_for("login"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        food_type = request.form.get("food_type", "Vegetables").strip()
        storage_type = request.form.get("storage", "Fridge").strip()
        purchase_date = validate_date(request.form.get("purchase_date", ""))
        temperature = request.form.get("temperature", "")
        humidity = request.form.get("humidity", "")

        if not (name and purchase_date and temperature and humidity):
            flash("All fields are required", "danger")
            return render_template("add_item.html")

        try:
            temperature = float(temperature)
            humidity = float(humidity)
        except ValueError:
            flash("Temperature and humidity must be numbers", "danger")
            return render_template("add_item.html")

        days_since_purchase = (datetime.now().date() - purchase_date).days
        predicted_days = predict_days(
            model_pipeline,
            food_type,
            storage_type,
            temperature,
            humidity,
            max(days_since_purchase, 0),
        )
        expiry_date = purchase_date + timedelta(days=predicted_days)
        status = classify_status((expiry_date - datetime.now().date()).days)

        conn = get_db_connection()
        conn.execute(
            "INSERT INTO items (user_id, name, food_type, storage_type, purchase_date, temperature, humidity, predicted_days, expiry_date, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                user["id"],
                name,
                food_type,
                storage_type,
                purchase_date.isoformat(),
                temperature,
                humidity,
                predicted_days,
                expiry_date.isoformat(),
                status,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        conn.close()
        flash(f"Added {name}: predicted {predicted_days} days (", "success")
        return redirect(url_for("dashboard"))
    return render_template("add_item.html")


@app.route("/update/<int:item_id>", methods=["GET", "POST"])
def update_item(item_id):
    user = get_user()
    if not user:
        return redirect(url_for("login"))
    conn = get_db_connection()
    item = conn.execute("SELECT * FROM items WHERE id = ? AND user_id = ?", (item_id, user["id"])).fetchone()
    if not item:
        conn.close()
        abort(404)
    if request.method == "POST":
        name = request.form.get("name", item["name"]).strip()
        food_type = request.form.get("food_type", item["food_type"]).strip()
        storage_type = request.form.get("storage", item["storage_type"]).strip()
        purchase_date = validate_date(request.form.get("purchase_date", item["purchase_date"]))
        temperature = request.form.get("temperature", item["temperature"])
        humidity = request.form.get("humidity", item["humidity"])
        try:
            temperature = float(temperature)
            humidity = float(humidity)
        except ValueError:
            flash("Temperature/humidity must be numeric", "danger")
            return render_template("update_item.html", item=item)
        days_since_purchase = (datetime.now().date() - purchase_date).days
        predicted_days = predict_days(
            model_pipeline,
            food_type,
            storage_type,
            temperature,
            humidity,
            max(days_since_purchase, 0),
        )
        expiry_date = purchase_date + timedelta(days=predicted_days)
        status = classify_status((expiry_date - datetime.now().date()).days)
        conn.execute(
            "UPDATE items SET name = ?, food_type = ?, storage_type = ?, purchase_date = ?, temperature = ?, humidity = ?, predicted_days = ?, expiry_date = ?, status = ? WHERE id = ? AND user_id = ?",
            (name, food_type, storage_type, purchase_date.isoformat(), temperature, humidity, predicted_days, expiry_date.isoformat(), status, item_id, user["id"]),
        )
        conn.commit()
        conn.close()
        flash("Item updated successfully", "success")
        return redirect(url_for("dashboard"))
    conn.close()
    return render_template("update_item.html", item=item)


@app.route("/delete/<int:item_id>")
def delete_item(item_id):
    user = get_user()
    if not user:
        return redirect(url_for("login"))
    conn = get_db_connection()
    conn.execute("DELETE FROM items WHERE id = ? AND user_id = ?", (item_id, user["id"]))
    conn.commit()
    conn.close()
    flash("Item removed", "info")
    return redirect(url_for("dashboard"))


@app.route("/recipes", methods=["GET", "POST"])
def recipes():
    user = get_user()
    if not user:
        return redirect(url_for("login"))
    ingredients = ""
    suggestions = []
    if request.method == "POST":
        ingredients = request.form.get("ingredients", "").strip()
        suggestions = recipe_engine.suggest(ingredients)
    return render_template("recipes.html", suggestions=suggestions, ingredients=ingredients)


@app.route("/recipe/<path:recipe_name>")
def recipe_detail(recipe_name):
    user = get_user()
    if not user:
        return redirect(url_for("login"))

    # find recipe by name (case-insensitive)
    recipe = None
    for r in recipe_engine.recipes:
        if r["name"].lower() == recipe_name.lower():
            recipe = r
            break

    if recipe is None:
        flash("Recipe not found", "warning")
        return redirect(url_for("recipes"))

    return render_template("recipe_detail.html", recipe=recipe)


# API endpoints

def api_auth_required():
    auth = request.authorization
    if not auth:
        return None
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ? AND password = ?", (auth.username, auth.password)).fetchone()
    conn.close()
    return user


@app.route("/api/items", methods=["GET", "POST"])
def api_items():
    user = api_auth_required()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    conn = get_db_connection()
    if request.method == "GET":
        rows = conn.execute("SELECT * FROM items WHERE user_id = ?", (user["id"],)).fetchall()
        conn.close()
        return jsonify([dict(r) for r in rows])
    data = request.json or {}
    required = ["name", "food_type", "storage_type", "purchase_date", "temperature", "humidity"]
    if not all(x in data for x in required):
        return jsonify({"error": "Missing fields"}), 400
    purchase_date = validate_date(data["purchase_date"])
    if not purchase_date:
        return jsonify({"error": "Invalid date"}), 400
    predicted_days = predict_days(
        model_pipeline,
        data["food_type"],
        data["storage_type"],
        float(data["temperature"]),
        float(data["humidity"]),
        0,
    )
    expiry_date = purchase_date + timedelta(days=predicted_days)
    status = classify_status((expiry_date - datetime.now().date()).days)
    conn.execute(
        "INSERT INTO items (user_id, name, food_type, storage_type, purchase_date, temperature, humidity, predicted_days, expiry_date, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (user["id"], data["name"], data["food_type"], data["storage_type"], purchase_date.isoformat(), float(data["temperature"]), float(data["humidity"]), predicted_days, expiry_date.isoformat(), status, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()
    return jsonify({"predicted_days": predicted_days, "expiry_date": expiry_date.isoformat(), "status": status}), 201


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json or {}
    required = ["food_type", "storage_type", "temperature", "humidity", "purchase_age_days"]
    if not all(x in data for x in required):
        return jsonify({"error": "Missing fields"}), 400
    predicted_days = predict_days(
        model_pipeline,
        data["food_type"],
        data["storage_type"],
        float(data["temperature"]),
        float(data["humidity"]),
        int(data["purchase_age_days"]),
    )
    return jsonify({"predicted_days": predicted_days}), 200


@app.route("/api/analytics", methods=["GET"])
def api_analytics():
    user = api_auth_required()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    conn = get_db_connection()
    rows = conn.execute("SELECT status, COUNT(*) AS count FROM items WHERE user_id = ? GROUP BY status", (user["id"],)).fetchall()
    conn.close()
    return jsonify({row["status"]: row["count"] for row in rows})


if __name__ == "__main__":
    init_db()
    app.run(debug=True)
