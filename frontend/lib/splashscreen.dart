import 'package:flutter/material.dart';
import 'package:animated_text_kit/animated_text_kit.dart';
import 'package:google_fonts/google_fonts.dart';
import 'dart:async';
import 'welcome.dart';

class Splash extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'My App',
      home: SplashScreen(),
    );
  }
}

class SplashScreen extends StatefulWidget {
  @override
  _SplashScreenState createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with TickerProviderStateMixin {
  late AnimationController _animationController;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      vsync: this,
      duration: Duration(seconds: 2),
    );

    // Start animation after 1 second delay
    Timer(Duration(seconds: 1), () {
      _animationController.forward();
    });

    // Navigate after animation completes (4 seconds)
    Timer(Duration(seconds: 2), () {
      Navigator.pushReplacement(
        context,
        PageRouteBuilder(
          pageBuilder: (context, animation, secondaryAnimation) => WELCOME(),
          transitionsBuilder: (context, animation, secondaryAnimation, child) {
            return FadeTransition(
              opacity: animation,
              child: child,
            );
          },
        ),
      );
    });
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            AnimatedTextKit(
              animatedTexts: [
                TypewriterAnimatedText(
                  'REACO',
                  textStyle: GoogleFonts.gabarito(
                    fontSize: 48,
                    fontWeight: FontWeight.bold,
                  ),
                  speed: Duration(milliseconds: 300),
                  textAlign: TextAlign.center,
                ),
              ],
              isRepeatingAnimation: false,
              repeatForever: false,
            ),
            const SizedBox(height: 80),
            AnimatedBuilder(
              animation: _animationController,
              builder: (context, child) {
                return Image.asset(
                  'images/loading.gif',
                  height: 200,
                  filterQuality: FilterQuality.high,
                );
              },
            ),
          ],
        ),
      ),
    );
  }
}
