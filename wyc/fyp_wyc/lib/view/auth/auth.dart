import 'package:flutter/material.dart';
import 'package:fyp_wyc/view/auth/login.dart';
import 'package:fyp_wyc/view/auth/sign_up.dart';

class AuthPage extends StatefulWidget {
  const AuthPage({super.key});

  @override
  State<AuthPage> createState() => _AuthPageState();
}

class _AuthPageState extends State<AuthPage> {
  late Widget _currentPage;
  late Widget _loginPage;
  late Widget _signUpPage;

  @override
  void initState() {
    super.initState();

    // set the pages
    _loginPage = LoginPage(
      onSignUpClicked: () {
        _setCurrentPage(_signUpPage);
      },
    );
    _signUpPage = SignUpPage(
      onLoginClicked: () {
        _setCurrentPage(_loginPage);
      },
    );

    _currentPage = _loginPage;
  }

  void _setCurrentPage(Widget page) {
    setState(() {
      _currentPage = page;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text('Welcome Back!',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
              ),
              SizedBox(height: 10),
              Text('Please enter your account here',
                style: TextStyle(
                  fontSize: 16,
                  color: Colors.grey,
                ),
              ),
              SizedBox(height: 30),
              _currentPage,
            ],
          ),
        ),
      ),
    );
  }
}