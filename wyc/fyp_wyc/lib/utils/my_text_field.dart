import 'package:flutter/material.dart';

class MyTextField extends StatelessWidget {
  final TextEditingController controller;
  final Icon? icon;
  final String hintText;
  final bool isPassword;
  final bool isVisible;
  final Function(String)? onChanged;
  final Color? backgroundColor;
  final bool borderDisplay;

  const MyTextField({
    super.key,
    required this.controller,
    this.icon,
    required this.hintText,
    this.isPassword = false,
    this.isVisible = true,
    this.onChanged,
    this.backgroundColor,
    this.borderDisplay = true,
  });

  @override
  Widget build(BuildContext context) {
    return TextFormField(
      controller: controller,
      decoration: InputDecoration(
        prefixIcon: icon,
        suffixIcon: isPassword ? IconButton(
          onPressed: () {
            onChanged?.call(controller.text);
          },
          icon: Icon(isVisible ? Icons.visibility : Icons.visibility_off),
        ) : null,
        hintText: hintText,
        filled: backgroundColor != null,
        fillColor: backgroundColor,
        border: borderDisplay ? OutlineInputBorder(
          borderRadius: BorderRadius.circular(25),
          borderSide: BorderSide(color: Colors.grey),
        ) : InputBorder.none,
        enabledBorder: !borderDisplay ? OutlineInputBorder(
          borderRadius: BorderRadius.circular(25),
          borderSide: BorderSide.none,
        ) : null,
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(25),
          borderSide: borderDisplay ? BorderSide(color: Colors.grey[700]!) : BorderSide.none,
        ),
        hintStyle: TextStyle(color: Colors.grey[400]),
        labelStyle: TextStyle(color: Colors.grey[800]),
      ),
      obscureText: !isVisible,
      style: TextStyle(color: Colors.grey[800]),
    );
  }
}