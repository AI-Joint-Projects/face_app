// ignore_for_file: prefer_const_constructors

import 'package:flutter/material.dart';
import 'package:reaco/create_events.dart';
import 'package:reaco/events_joined.dart';
import 'package:reaco/join_events.dart';
import 'package:reaco/my_events.dart';
import 'package:reaco/profile.dart';

class DashBoard extends StatefulWidget {
  const DashBoard({super.key, required this.userData});
  final Map<String, dynamic> userData;

  @override
  State<DashBoard> createState() => _DashBoardState();
}

class _DashBoardState extends State<DashBoard> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
         backgroundColor: Colors.transparent,
        title: Text(
          'Dashboard',
          style: TextStyle(
            fontFamily: 'Montserrat',
            fontWeight: FontWeight.bold,
            fontSize: 25,
            color: Colors.black,
          ),
        ),
       
        elevation: 0, // Removes the shadow from the AppBar
        centerTitle: true, // Centers the title
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: GridView.count(
          crossAxisCount: 2,
          crossAxisSpacing: 16,
          mainAxisSpacing: 16,
          childAspectRatio: 1.2,
          children: <Widget>[
            _buildButton(
              context: context,
              icon: Icons.person,
              label: 'Profile',
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => ProfileScreen(userData: widget.userData)),
                );
              },
            ),
            _buildButton(
              context: context,
              icon: Icons.event,
              label: 'My Events',
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => MyEventsScreen(userData: widget.userData)),
                );
              },
            ),
            _buildButton(
              context: context,
              icon: Icons.group,
              label: 'Events Joined',
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => EventsJoinedScreen(userData: widget.userData)),
                );
              },
            ),
            _buildButton(
              context: context,
              icon: Icons.add_circle,
              label: 'Create Event',
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => CreateEventScreen(userData: widget.userData)),
                );
              },
            ),
            _buildButton(
              context: context,
              icon: Icons.search,
              label: 'Join Events',
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => JoinEventsScreen(userData: widget.userData)),
                );
              },
            ),
            // Removed the empty container for a cleaner look
          ],
        ),
      ),
    );
  }

  Widget _buildButton({
    required BuildContext context,
    required IconData icon,
    required String label,
    required VoidCallback onPressed,
  }) {
    return ElevatedButton(
      onPressed: onPressed,
      style: ElevatedButton.styleFrom(
        backgroundColor: Colors.grey[850],
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(20), // More rounded corners
        ),
        padding: EdgeInsets.symmetric(vertical: 24, horizontal: 24), // Increased padding
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            icon,
            color: Colors.white,
            size: 40, // Larger icon
          ),
          SizedBox(height: 8),
          Text(
            label,
            style: TextStyle(
              fontFamily: 'Montserrat',
              color: Colors.white,
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }
}