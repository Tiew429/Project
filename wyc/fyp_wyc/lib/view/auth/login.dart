import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import 'package:fyp_wyc/utils/my_button.dart';
import 'package:fyp_wyc/utils/my_text_field.dart';
import 'package:go_router/go_router.dart';

class LoginPage extends StatefulWidget {
  final VoidCallback onSignUpClicked;

  const LoginPage({
    super.key, 
    required this.onSignUpClicked,
  });

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final _formKey = GlobalKey<FormState>();
  final _emailOrPhoneController = TextEditingController();
  final _passwordController = TextEditingController();
  bool isPasswordVisible = false;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Form(
          key: _formKey,
          child: Column(
            children: [
              // email or phone number text field
              MyTextField(
                controller: _emailOrPhoneController,
                icon: Icon(Icons.email),
                hintText: 'Email or phone number',
              ),
              SizedBox(height: 15),
              // password text field
              MyTextField(
                controller: _passwordController,
                icon: Icon(Icons.lock),
                hintText: 'Password',
                isPassword: true,
                isVisible: isPasswordVisible,
                onChanged: (value) {
                  setState(() {
                    isPasswordVisible = !isPasswordVisible;
                  });
                },
              ),
            ],
          ),
        ),
        SizedBox(height: 20),
        // forgot password
        Row(
          children: [
            Spacer(),
            Text('Forgot password?',
              style: TextStyle(
                fontSize: 16,
              ),
            ),
          ]
        ),
        SizedBox(height: 40),
        // login button
        MyButton(
          onPressed: () {},
          text: 'Login',
          backgroundColor: const Color.fromARGB(255, 26, 218, 128),
        ),
        SizedBox(height: 30),
        Text('Or continue with',
          style: TextStyle(
            fontSize: 16,
            color: Colors.grey,
          ),
        ),
        SizedBox(height: 30),
        // google button
        MyButton(
          onPressed: () {},
          text: 'Google',
          backgroundColor: const Color.fromARGB(255, 244, 81, 63),
          icon: Icon(FontAwesomeIcons.google,
            color: Colors.white,
          ),
        ),
        SizedBox(height: 15),
        // continue as guest
        MyButton(
          onPressed: () => context.go('/dashboard'),
          text: 'Continue as guest',
          backgroundColor: Colors.grey,
        ),
        SizedBox(height: 15),
        // sign up text
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text('Don\'t have any account?'),
            TextButton(
              onPressed: widget.onSignUpClicked,
              child: Text('Sign up',
                style: TextStyle(
                  color: Color.fromARGB(255, 26, 218, 128),
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ],
        ),
      ],
    );
  }
}