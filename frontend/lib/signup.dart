import 'package:flutter/material.dart';
import 'package:http/http.dart';
import 'package:reaco/login.dart';
import 'package:reaco/upload_images.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

import 'package:reaco/verificaton.dart';

class SignUpScreen extends StatelessWidget {
  const SignUpScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Sign Up',
          style: TextStyle(
            fontFamily: 'Montserrat',
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
      body: const Center(
        child: RegistrationScreen(),
      ),
    );
  }
}

class RegistrationScreen extends StatefulWidget {
  const RegistrationScreen({super.key});

  @override
  _RegistrationScreenState createState() => _RegistrationScreenState();
}

class _RegistrationScreenState extends State<RegistrationScreen> {
  final _formKey = GlobalKey<FormState>();
  final _nameController = TextEditingController();
  final _emailController = TextEditingController();
  final _phoneController = TextEditingController();
  final _passwordController = TextEditingController();
  final _confirmPasswordController = TextEditingController();
  final _birthdateController = TextEditingController();
  bool _isLoading = false;

  @override
  void dispose() {
    _nameController.dispose();
    _emailController.dispose();
    _phoneController.dispose();
    _passwordController.dispose();
    _confirmPasswordController.dispose();
    _birthdateController.dispose();
    super.dispose();
  }

  InputDecoration _getInputDecoration(String label) {
    return InputDecoration(
      labelText: label,
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(40),
      ),
      contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
    );
  }

  Future<void> _submitForm() async {
    if (_formKey.currentState!.validate()) {
      setState(() {
        _isLoading = true; // Show progress indicator
      });

      // Create user object from form data
      final user = {
        'username': _nameController.text,
        'email': _emailController.text,
        'password': _passwordController.text,
        'phone_number': _phoneController.text,
        'first_name': _nameController.text.split(' ')[0],
        'last_name': _nameController.text.split(' ').length > 1
            ? _nameController.text.split(' ')[1]
            : '',
        'birthdate': _birthdateController.text,
        'account_type': 'basic',
        'is_verified': false,
        
      };

      // Send POST request to backend API for registration
      final registrationResponse = await http.post(
        Uri.parse('https://x200lnxp-8000.inc1.devtunnels.ms/users/register'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(user),
      );

      // Handle registration response
      if (registrationResponse.statusCode == 201) {
        // Registration successful, now try to log in
        Response loginResponse = await http.post(
          Uri.parse('https://x200lnxp-8000.inc1.devtunnels.ms/login'),
          body: {
            'username': user['email'],
            'password': user['password'],
          },
        );

        if (loginResponse.statusCode == 200) {
          // Login successful
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Registration and Login successful!'),
            ),
          );
          // Decode user data from login response
          final Map<String, dynamic> userData = jsonDecode(loginResponse.body);
          // Navigate to the UploadScreen
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => Verification(userData: userData),
            ),
          );
        } else {
          // Login failed
          print('Login failed: ${loginResponse.body}');
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Login failed. Please try again.'),
            ),
          );
        }
      } else {
        // Registration failed
        print('Registration failed: ${registrationResponse.body}');
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Registration failed. Please try again.'),
          ),
        );
      }

      setState(() {
        _isLoading = false; // Hide progress indicator
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        Padding(
          padding: const EdgeInsets.all(16.0),
          child: SingleChildScrollView(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Text(
                  'WELCOME',
                  style: TextStyle(
                    fontSize: 24.0,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 16.0),
                const Text(
                  'Lets start with registration',
                  style: TextStyle(fontSize: 16.0),
                ),
                const SizedBox(height: 32.0),
                Form(
                  key: _formKey,
                  child: Column(
                    children: [
                      TextFormField(
                        controller: _nameController,
                        decoration: _getInputDecoration('Enter your full name'),
                        validator: (value) {
                          if (value == null || value.isEmpty) {
                            return 'Please enter your name';
                          }
                          return null;
                        },
                      ),
                      const SizedBox(height: 16.0),
                      TextFormField(
                        controller: _emailController,
                        decoration: _getInputDecoration('Enter your email'),
                        validator: (value) {
                          if (value == null || value.isEmpty) {
                            return 'Please enter your email';
                          }
                          if (!value.contains('@') || !value.contains('.')) {
                            return 'Please enter a valid email address';
                          }
                          return null;
                        },
                      ),
                      const SizedBox(height: 16.0),
                      TextFormField(
                        controller: _phoneController,
                        decoration: _getInputDecoration('Phone number'),
                        keyboardType: TextInputType.phone,
                      ),
                      const SizedBox(height: 16.0),
                      TextFormField(
                        controller: _passwordController,
                        obscureText: true,
                        decoration: _getInputDecoration('Enter password'),
                        validator: (value) {
                          if (value == null || value.isEmpty) {
                            return 'Please enter a password';
                          }
                          return null;
                        },
                      ),
                      const SizedBox(height: 16.0),
                      TextFormField(
                        controller: _confirmPasswordController,
                        obscureText: true,
                        decoration: _getInputDecoration('Confirm password'),
                        validator: (value) {
                          if (value != _passwordController.text) {
                            return 'Passwords do not match';
                          }
                          return null;
                        },
                      ),
                      const SizedBox(height: 16.0),
                      TextFormField(
                        controller: _birthdateController,
                        decoration: _getInputDecoration(
                            'Enter your birthdate (YYYY-MM-DD)'),
                        keyboardType: TextInputType.datetime,
                        validator: (value) {
                          if (value == null || value.isEmpty) {
                            return 'Please enter your birthdate';
                          }
                          // Add validation for date format
                          return null;
                        },
                      ),
                      const SizedBox(height: 32.0),
                      ElevatedButton(
                        onPressed: _submitForm,
                        style: ElevatedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(horizontal: 40.0),
                          backgroundColor: Colors.black,
                          foregroundColor: Colors.white,
                        ),
                        child: const Text('REGISTER'),
                      ),
                      const SizedBox(height: 16.0),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          const Text('Already have an account? '),
                          TextButton(
                            onPressed: () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                  builder: (context) => const SCREENN(),
                                ),
                              );
                            },
                            child: const Text('Sign in'),
                          ),
                        ],
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
        if (_isLoading)
          ModalBarrier(
            dismissible: false,
            color: Colors.black.withOpacity(0.5),
          ),
        if (_isLoading)
          const Center(
            child: CircularProgressIndicator(
              color: Colors.white,
            ),
          ),
      ],
    );
  }
}
