import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:reaco/upload_images.dart';

class Verification extends StatefulWidget {
  const Verification({super.key, required this.userData});
  final Map<String, dynamic> userData;

  @override
  _VerificationState createState() => _VerificationState();
}

class _VerificationState extends State<Verification> {
  final TextEditingController _otpController = TextEditingController();
  bool _otpSent = false;
  bool _isLoading = false;

  void _sendOtp() async {
    setState(() {
      _isLoading = true;
    });

    final authToken = widget.userData['access_token'];
    final url = Uri.parse('https://x200lnxp-8000.inc1.devtunnels.ms/users/generate-otp');
    final response = await http.post(
      url,
      headers: {'Authorization': 'Bearer $authToken'},
    );

    if (response.statusCode == 200) {
      setState(() {
        _otpSent = true;
      });
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('OTP sent to your email.')));
    } else {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Failed to send OTP.')));
    }

    setState(() {
      _isLoading = false;
    });
  }

  void _verifyOtp() async {
    setState(() {
      _isLoading = true;
    });

    final authToken = widget.userData['access_token'];
    String otp = _otpController.text;
    final url = Uri.parse('https://x200lnxp-8000.inc1.devtunnels.ms/users/verify-otp').replace(queryParameters: {'user_otp': otp});
    final response = await http.post(
      url,
      headers: {
        'Authorization': 'Bearer $authToken',
        'Content-Type': 'application/json',
      },
    );

    if (response.statusCode == 202) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('OTP verified.')));
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (context) => UploadScreen(userData: widget.userData),
        ),
      );
    } else {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Failed to verify OTP.')));
    }

    setState(() {
      _isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    String username = widget.userData['name'] ?? 'User';

    return Scaffold(
      appBar: AppBar(
        title: Text('Verification'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              'Welcome, $username',
              style: TextStyle(fontSize: 24),
            ),
            SizedBox(height: 20),
            Text('Your account is unverified. To verify your account:'),
            SizedBox(height: 20),
            if (_isLoading)
              CircularProgressIndicator()
            else if (!_otpSent)
              ElevatedButton(
                onPressed: _sendOtp,
                child: Text('Send OTP'),
              )
            else
              Column(
                children: [
                  TextField(
                    controller: _otpController,
                    keyboardType: TextInputType.number,
                    decoration: InputDecoration(
                      labelText: 'Enter 6-digit OTP',
                      border: OutlineInputBorder(),
                    ),
                    maxLength: 6,
                  ),
                  SizedBox(height: 20),
                  ElevatedButton(
                    onPressed: _verifyOtp,
                    child: Text('Verify OTP'),
                  ),
                ],
              ),
          ],
        ),
      ),
    );
  }
}
