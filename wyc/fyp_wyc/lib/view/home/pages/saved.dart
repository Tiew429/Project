import 'package:flutter/material.dart';
import 'package:fyp_wyc/utils/my_recipe_box.dart';

class SavedPage extends StatefulWidget {
  const SavedPage({super.key});

  @override
  State<SavedPage> createState() => _SavedPageState();
}

class _SavedPageState extends State<SavedPage> {
  bool isSaved = true; // for testing purposes

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // title
        Row(
          children: [
            Text('My Saved Recipes',
              style: TextStyle(
                fontSize: 24,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
        // saved recipes grid
        Expanded(
          child: GridView.builder(
            padding: EdgeInsets.only(top: 16.0),
            gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: 2,
              childAspectRatio: 0.75,
              crossAxisSpacing: 16,
              mainAxisSpacing: 16,
            ),
            itemCount: 12, // Should be replaced with actual recipe count
            itemBuilder: (context, index) {
              return MyRecipeBox(
                imageUrl: "https://source.unsplash.com/random/300x200?food&sig=$index",
                title: "Recipe ${index + 1}",
                cookTime: 30,
                isSaved: isSaved,
                onSave: () {
                  setState(() {
                    isSaved = !isSaved;
                  });
                },
                onTap: () {},
              );
            },
          ),
        ),
      ],
    );
  }
}