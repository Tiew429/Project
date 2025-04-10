import 'package:flutter/material.dart';
import 'package:fyp_wyc/utils/my_text_field.dart';

class AddRecipePage extends StatefulWidget {
  const AddRecipePage({super.key});

  @override
  State<AddRecipePage> createState() => _AddRecipePageState();
}

class _AddRecipePageState extends State<AddRecipePage> {
  final TextEditingController _titleController = TextEditingController();
  final TextEditingController _descriptionController = TextEditingController();
  final TextEditingController _instructionsController = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text('Add Recipe',
          style: TextStyle(
            fontSize: 24,
            fontWeight: FontWeight.bold,
          ),
        ),
        SizedBox(height: 20),
        // recipe title
        Text('Recipe Title',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
        ),
        SizedBox(height: 10),
        SizedBox(
          width: MediaQuery.of(context).size.width * 0.7,
          child: MyTextField(
            controller: _titleController,
            hintText: 'Enter recipe title',
            borderDisplay: false,
            backgroundColor: const Color.fromARGB(255, 236, 237, 248),
          ),
        ),
        SizedBox(height: 10),
        // recipe cover image
        Text('Recipe Cover Image',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.bold,
          ),
        ),
        
      ],
    );
  }
}