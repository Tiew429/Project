import 'package:flutter/material.dart';

class MyButton extends StatelessWidget {
  final VoidCallback onPressed;
  final String text;
  final Color backgroundColor;
  final Icon? icon;
  final double? width;
  final double? height;
  final double? borderRadius;

  const MyButton({
    super.key,
    required this.onPressed,
    required this.text,
    this.backgroundColor = const Color.fromARGB(255, 26, 218, 128),
    this.icon,
    this.width,
    this.height,
    this.borderRadius,
  });

  @override
  Widget build(BuildContext context) {
    final screenSize = MediaQuery.of(context).size;

    return ElevatedButton(
      onPressed: onPressed,
      style: ElevatedButton.styleFrom(
        fixedSize: Size(width ?? screenSize.width, height ?? 50),
        backgroundColor: backgroundColor,
        padding: EdgeInsets.symmetric(horizontal: 40, vertical: 15),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(borderRadius ?? 30),
        ),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          icon ?? SizedBox.shrink(),
          SizedBox(width: 10),
          Text(text,
            style: TextStyle(
              color: Colors.white,
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
          ),
        ]
      ),
    );
  }
}