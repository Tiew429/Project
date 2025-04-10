import 'package:fyp_wyc/model/ingredient.dart';

class Recipe {
  final String recipeID;
  final String recipeName;
  final String description;
  final String imageUrl;
  final String category;
  final List<Ingredient> ingredients;
  final List<String> steps;
  final String authorID;

  const Recipe({
    required this.recipeID,
    required this.recipeName,
    required this.description,
    required this.imageUrl,
    required this.category,
    required this.ingredients,
    required this.steps,
    required this.authorID,
  });
}
