import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

import 'package:reaco/create_events.dart';
import 'package:reaco/event_images.dart';
import 'package:reaco/join_events.dart';

class EventsJoinedScreen extends StatefulWidget {
  const EventsJoinedScreen({super.key, required this.userData});
  final Map<String, dynamic> userData;

  @override
  _EventsJoinedScreenState createState() => _EventsJoinedScreenState();
}

class _EventsJoinedScreenState extends State<EventsJoinedScreen> {
  List<Map<String, dynamic>> eventsJoined = [];
  bool isLoading = true;
  String? errorMessage;

  @override
  void initState() {
    super.initState();
    _fetchJoinedEvents();
  }

  Future<void> _fetchJoinedEvents() async {
    try {
      final response = await http.get(
        Uri.parse('https://x200lnxp-8000.inc1.devtunnels.ms/events/joined-events'),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer ${widget.userData['access_token']}',
        },
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data is Map<String, dynamic> && data.containsKey('joined_events')) {
          setState(() {
            eventsJoined = List<Map<String, dynamic>>.from(data['joined_events']);
            isLoading = false;
          });
        } else {
          setState(() {
            errorMessage = 'Invalid response format';
            isLoading = false;
          });
        }
      } else {
        setState(() {
          errorMessage = 'Failed to load events: ${response.statusCode}';
          isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        errorMessage = 'Failed to load events: $e';
        isLoading = false;
      });
    }
  }

  Future<void> _leaveEvent(String eventUrl) async {
    final url = Uri.parse(
      'https://x200lnxp-8000.inc1.devtunnels.ms/profile/leave-event',
    ).replace(queryParameters: {'event_id': eventUrl});

    try {
      final response = await http.put(url, headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ${widget.userData['access_token']}',
      });

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['message'] == "event :$eventUrl left successfully") {
          setState(() {
            eventsJoined.removeWhere((event) => event['event_url'] == eventUrl);
          });
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Left event successfully')),
          );
        } else {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Failed to leave event: ${data['message']}')),
          );
        }
      } else {
        print('Error: ${response.statusCode}, ${response.body}');
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to leave event: ${response.body}')),
        );
      }
    } catch (e) {
      print('Error: $e');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to leave event: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Events Joined'),
      ),
      body: isLoading
          ? const Center(child: CircularProgressIndicator())
          : errorMessage != null
              ? Center(
                  child: Text(errorMessage!),
                )
              : eventsJoined.isEmpty
                  ? Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          const Text('No Events Joined'),
                          const SizedBox(height: 20),
                          ElevatedButton(
                            onPressed: () {
                              Navigator.push(
                                context,
                                MaterialPageRoute(
                                  builder: (context) =>
                                      JoinEventsScreen(userData: widget.userData),
                                ),
                              );
                            },
                            style: ElevatedButton.styleFrom(
                              backgroundColor: Colors.black,
                              foregroundColor: Colors.white,
                            ),
                            child: const Text('Join an Event'),
                          ),
                        ],
                      ),
                    )
                  : ListView.builder(
                      itemCount: eventsJoined.length,
                      itemBuilder: (context, index) {
                        final event = eventsJoined[index];
                        final eventUrl = event['event_url']; // Assuming 'event_url' exists
                        return Card(
                          child: ListTile(
                            leading: CircleAvatar(
                              backgroundColor: Colors.black,
                              child: Text(
                                '${index + 1}',
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ),
                            title: Text(
                              event['event_name'],
                              style: const TextStyle(
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            subtitle: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text('URL: $eventUrl'),
                                Text('Completed: ${event['is_complete']}'),
                              ],
                            ),
                            trailing: Row(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                IconButton(
                                  icon: const Icon(Icons.image, color: Colors.black),
                                  onPressed: () {
                                    Navigator.push(
                                      context,
                                      MaterialPageRoute(
                                        builder: (context) => EventImagesScreen(
                                          eventUrl: eventUrl,
                                          userData: widget.userData, // Pass userData here
                                        ),
                                      ),
                                    );
                                  },
                                ),
                                IconButton(
                                  icon: const Icon(Icons.exit_to_app, color: Colors.red),
                                  onPressed: () {
                                    _leaveEvent(eventUrl); // Leave event on button press
                                  },
                                ),
                              ],
                            ),
                          ),
                        );
                      },
                    ),
    );
  }
}
