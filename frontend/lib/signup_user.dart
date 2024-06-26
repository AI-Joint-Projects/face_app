// import 'package:flutter/material.dart';








// class UserPage extends StatelessWidget {
//   const UserPage({super.key});

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(
//         title: const Text('Sign Up'),
//       ),
//       body: const Center(
//         child: RegistrationScreen(),
//       ),
//     );
//   }
// }

// class RegistrationScreen extends StatefulWidget {
//   const RegistrationScreen({super.key});

//   @override
//   _RegistrationScreenState createState() => _RegistrationScreenState();
// }

// class _RegistrationScreenState extends State<RegistrationScreen> {
//   final _formKey = GlobalKey<FormState>();
//   final _nameController = TextEditingController();
//   final _emailController = TextEditingController();
//   final _phoneController = TextEditingController();
//   final _passwordController = TextEditingController();
//   final _confirmPasswordController = TextEditingController();

//   @override
//   void dispose() {
//     _nameController.dispose();
//     _emailController.dispose();
//     _phoneController.dispose();
//     _passwordController.dispose();
//     _confirmPasswordController.dispose();
//     super.dispose();
//   }

//   InputDecoration _getInputDecoration(String label) {
//     return InputDecoration(
//       labelText: label,
//       border: OutlineInputBorder(
//         borderRadius: BorderRadius.circular(40),
//       ),
//       contentPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
//     );
//   }

//   @override
//   Widget build(BuildContext context) {
//     return Padding(
//       padding: const EdgeInsets.all(16.0),
//       child: SingleChildScrollView(
//         child: Column(
//           mainAxisAlignment: MainAxisAlignment.center,
//           children: [
//             const Text(
//               'WELCOME',
//               style: TextStyle(
//                 fontSize: 24.0,
//                 fontWeight: FontWeight.bold,
//               ),
//             ),
//             const SizedBox(height: 16.0),
//             const Text(
//               'Lets start with registration',
//               style: TextStyle(fontSize: 16.0),
//             ),
//             const SizedBox(height: 32.0),
//             Form(
//               key: _formKey,
//               child: Column(
//                 children: [
//                   TextFormField(
//                     controller: _nameController,
//                     decoration: _getInputDecoration('Enter your full name'),
//                     validator: (value) {
//                       if (value == null || value.isEmpty) {
//                         return 'Please enter your name';
//                       }
//                       return null;
//                     },
//                   ),
//                   const SizedBox(height: 16.0),
//                   TextFormField(
//                     controller: _emailController,
//                     decoration: _getInputDecoration('Enter your email'),
//                     validator: (value) {
//                       if (value == null || value.isEmpty) {
//                         return 'Please enter your email';
//                       }
//                       return null;
//                     },
//                   ),
//                   const SizedBox(height: 16.0),
//                   TextFormField(
//                     controller: _phoneController,
//                     decoration: _getInputDecoration('Phone number'),
//                   ),
//                   const SizedBox(height: 16.0),
//                   TextFormField(
//                     controller: _passwordController,
//                     obscureText: true,
//                     decoration: _getInputDecoration('Enter password'),
//                     validator: (value) {
//                       if (value == null || value.isEmpty) {
//                         return 'Please enter a password';
//                       }
//                       return null;
//                     },
//                   ),
//                   const SizedBox(height: 16.0),
//                   TextFormField(
//                     controller: _confirmPasswordController,
//                     obscureText: true,
//                     decoration: _getInputDecoration('Confirm password'),
//                     validator: (value) {
//                       if (value != _passwordController.text) {
//                         return 'Passwords do not match';
//                       }
//                       return null;
//                     },
//                   ),
//                   const SizedBox(height: 32.0),




//                   //  Text('User'),
//                   // Text('Photographer'),

                

//                   const SizedBox(height: 16.0), // Add some space between the buttons




                 







//                   ElevatedButton(
//                     onPressed: () {
//                       if (_formKey.currentState!.validate()) {
//                         // Perform registration logic here
//                         print('Registration successful');
//                       }
//                     },
//                     style: ElevatedButton.styleFrom(
//                       padding: const EdgeInsets.symmetric(horizontal: 40.0),
//                       backgroundColor: Colors.black,
//                       foregroundColor: Colors.white,
//                     ),
//                     child: const Text('REGISTER'),
//                   ),





                 


//                   const SizedBox(height: 16.0),
//                   Row(
//                     mainAxisAlignment: MainAxisAlignment.center,
//                     children: [
//                       const Text('Already have an account? '),
//                       TextButton(
//                         onPressed: () {
                          
//                          // Perform registration logic here
//                         print('Registration successful');
//                         Navigator.of(context).pushReplacementNamed('/home'); // Navigates to the home screen
    
//                           // Navigate to sign-in screen
//                         },
//                         child: const Text('Sign in'),
//                       ),
//                     ],
//                   ),
//                 ],
//               ),
//             ),
//           ],
//         ),
//       ),
//     );
//   }
// }
