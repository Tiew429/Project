import 'package:flutter/material.dart';
import 'package:fyp_wyc/utils/my_avatar.dart';
import 'package:fyp_wyc/utils/my_recipe_box.dart';
import 'package:fyp_wyc/utils/my_search_bar.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final TextEditingController _searchController = TextEditingController();
  bool isSaved = false; // for testing purposes

  @override
  Widget build(BuildContext context) {
    final screenSize = MediaQuery.of(context).size;

    return Column(
      children: [
        // top appbar widgets
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            // top-left user avatar
            MyAvatar(),
            SizedBox(width: screenSize.width * 0.05),
            // top search bar
            SizedBox(
              width: screenSize.width * 0.7,
              child: MySearchBar(
                controller: _searchController,
                hintText: 'Search...',
                onChanged: (value) {},
              ),
            ),
          ],
        ),
        SizedBox(height: 10),
        // body widgets - Recipe Grid
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