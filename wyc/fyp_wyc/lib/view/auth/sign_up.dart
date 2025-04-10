import 'package:flutter/material.dart';
import 'package:fyp_wyc/utils/my_button.dart';
import 'package:fyp_wyc/utils/my_text_field.dart';

class SignUpPage extends StatefulWidget {
  final VoidCallback onLoginClicked;

  const SignUpPage({
    super.key,
    required this.onLoginClicked,
  });

  @override
  State<SignUpPage> createState() => _SignUpPageState();
}

class _SignUpPageState extends State<SignUpPage> {
  final _formKey = GlobalKey<FormState>();
  final _emailController = TextEditingController();
  final _passwordController = TextEditingController();
  bool isPasswordVisible = false;

  @override
  Widget build(BuildContext context) {
    return Form(
      key: _formKey,
      child: Column(
        children: [
          // email or phone number text field
          MyTextField(
            controller: _emailController,
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
          SizedBox(height: 30),
          Row(
            children: [
              Text('Your Password must contain: '),
            ],
          ),
          SizedBox(height: 20),
          // password requirements
          _buildPasswordRequirements('At least 6 characters'),
          _buildPasswordRequirements('Contains a number'),
          // sign up button
          MyButton(
            onPressed: () {}, 
            text: 'Sign up',
          ),
          // login text
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text('Already have an account?'),
              TextButton(
                onPressed: widget.onLoginClicked,
                child: Text('Login',
                  style: TextStyle(
                    color: Color.fromARGB(255, 26, 218, 128),
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),

            ]
          )
        ],
      ),
    );
  }

  Widget _buildPasswordRequirements(String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Row(
        children: [
          Icon(
            Icons.check_circle,
            color: Colors.green,
          ),
          SizedBox(width: 5),
          Text(text),
        ],
      ),
    );
  }
}