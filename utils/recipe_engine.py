import os
import csv
from difflib import SequenceMatcher


class RecipeEngine:
    def __init__(self):
        self.recipes = []
        self.load_recipes()

    def load_recipes(self):
        default_recipes = [
            {
                "name": "Vegetable Stir Fry",
                "ingredients": ["vegetables", "soy sauce", "garlic", "oil", "rice"],
                "steps": "Sauté vegetables with garlic, add soy sauce, serve with rice.",
            },
            {
                "name": "Fruit Smoothie",
                "ingredients": ["fruits", "milk", "yogurt", "honey", "ice"],
                "steps": "Blend all ingredients until smooth.",
            },
            {
                "name": "Chicken Rice Bowl",
                "ingredients": ["chicken", "rice", "spices", "veggies"],
                "steps": "Cook chicken and veggies, serve on rice.",
            },
        ]

        self.recipes = default_recipes
        recipes_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "recipes.csv")
        if os.path.exists(recipes_file):
            try:
                with open(recipes_file, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    csv_recipes = []
                    for row in reader:
                        ingredients = [x.strip().lower() for x in row.get("ingredients", "").split("|") if x.strip()]
                        if not row.get("name") or not ingredients:
                            continue
                        csv_recipes.append(
                            {
                                "name": row["name"],
                                "ingredients": ingredients,
                                "steps": row.get("steps", "No instructions provided."),
                            }
                        )
                    if csv_recipes:
                        self.recipes = csv_recipes
            except Exception:
                pass

    def _score(self, ingredient_text, recipe_ing):
        return SequenceMatcher(None, ingredient_text, recipe_ing).ratio()

    def suggest(self, ingredients_text):
        if not ingredients_text:
            return []

        search_items = [token.strip().lower() for token in ingredients_text.split(",") if token.strip()]
        matched_recipes = []

        for recipe in self.recipes:
            match_score = 0.0
            matched = False
            for search in search_items:
                for ingredient in recipe["ingredients"]:
                    if search in ingredient or ingredient in search:
                        matched = True
                        match_score += 1.0
                    else:
                        ratio = self._score(search, ingredient)
                        if ratio >= 0.65:
                            matched = True
                            match_score += ratio

            if matched:
                matched_recipes.append((match_score, recipe))

        matched_recipes.sort(key=lambda x: x[0], reverse=True)
        return [recipe for score, recipe in matched_recipes][:8]

