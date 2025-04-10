import 'package:flutter/material.dart';
import 'package:fyp_wyc/utils/my_avatar.dart';

class ProfilePage extends StatefulWidget {
  const ProfilePage({super.key});

  @override
  State<ProfilePage> createState() => _ProfilePageState();
}

class _ProfilePageState extends State<ProfilePage> {
  @override
  Widget build(BuildContext context) {
    final screenSize = MediaQuery.of(context).size;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        Padding(
          padding: const EdgeInsets.only(top: 16.0),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              SizedBox(
                width: screenSize.width * 0.6,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // user name
                    Text('Yan Cheng',
                      style: TextStyle(
                        fontSize: 24,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    SizedBox(height: 30),
                    // edit profile
                    Text('edit your profile',
                      style: TextStyle(
                        fontSize: 20,
                        decoration: TextDecoration.underline,
                      ),
                    ),
                  ],
                ),
              ),
              // user avatar
              Container(
                width: screenSize.width * 0.3,
                alignment: Alignment.center,
                child: MyAvatar(
                  radius: screenSize.width * 0.10,
                ),
              ),
            ],
          ),
        ),
        SizedBox(height: 50),
        // services and policy options
        Container(
          width: screenSize.width * 0.9,
          decoration: BoxDecoration(
            color: const Color.fromARGB(255, 236, 237, 248),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20.0, vertical: 10.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                SizedBox(height: 10),
                Text('Contact Us'),
                SizedBox(height: 20),
                Text('Terms of Service'),
                SizedBox(height: 20),
                Text('Privacy Policy'),
                SizedBox(height: 20),
                Text('Log Out'),
                SizedBox(height: 50), // just a dummy space
              ],
            ),
          ),
        ),
        SizedBox(height: 50),
        // app version
        Text('Version 1.0'),
      ],
    );
  }
}
