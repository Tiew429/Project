import 'package:flutter/material.dart';

class MyAvatar extends StatelessWidget {
  final double? radius;
  final Image? image;
  final Icon? icon;

  const MyAvatar({
    super.key,
    this.radius,
    this.image,
    this.icon,
  });

  @override
  Widget build(BuildContext context) {
    final screenSize = MediaQuery.of(context).size;

    return CircleAvatar(
      radius: radius ?? screenSize.width * 0.07,
      backgroundColor: const Color.fromARGB(255, 236, 237, 248),
      child: image ?? (icon ?? Icon(Icons.person, color: Colors.grey[800])),
    );
  }
}