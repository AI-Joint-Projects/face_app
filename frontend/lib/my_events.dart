import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

import 'package:reaco/create_events.dart';
import 'package:reaco/user_image_upload.dart';

class MyEventsScreen extends StatefulWidget {
  const MyEventsScreen({super.key, required this.userData});
  final Map<String, dynamic> userData;

  @override
  _MyEventsScreenState createState() => _MyEventsScreenState();
}

class _MyEventsScreenState extends State<MyEventsScreen> {
  List<dynamic> myEvents = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _fetchEvents();
  }

  Future<void> _fetchEvents() async {
    final url = Uri.parse(
        'https://x200lnxp-8000.inc1.devtunnels.ms/events/created-events');

    try {
      final response = await http.get(url, headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ${widget.userData['access_token']}'
      });

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          myEvents = data['created_events'] ?? [];
          _isLoading = false;
        });
      } else {
        print('Error: ${response.statusCode}, ${response.body}');
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to load events: ${response.body}')),
        );
      }
    } catch (e) {
      setState(() {
        _isLoading = false;
      });
      print('Error: $e');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to load events: $e')),
      );
    }
  }

  Future<void> _deleteEvent(String eventUrl) async {
    final url = Uri.parse(
        'https://x200lnxp-8000.inc1.devtunnels.ms/profile/delete-event')
      .replace(queryParameters: {'event_url': eventUrl});

    try {
      final response = await http.delete(url, headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ${widget.userData['access_token']}'
      });

      if (response.statusCode == 200) {
        // Event deleted successfully
        print('Event deleted successfully!');
        setState(() {
          myEvents.removeWhere((event) => event['event_url'] == eventUrl);
        });
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Event deleted successfully')),
        );
      } else {
        print('Error: ${response.statusCode}, ${response.body}');
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to delete event: ${response.body}')),
        );
      }
    } catch (e) {
      print('Error: $e');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to delete event: $e')),
      );
    }
  }

  Future<void> _markEventAsComplete(String eventUrl) async {
    final url = Uri.parse(
        'https://x200lnxp-8000.inc1.devtunnels.ms/events/complete-event').replace(queryParameters: {
      'event_url': eventUrl,
      'complete': 'true'
    });


    print('URL: $url');
   

    try {
      final response = await http.put(url, headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ${widget.userData['access_token']}'
      });

      print('Response status: ${response.statusCode}');
      print('Response body: ${response.body}');

      if (response.statusCode == 200) {
        // Event marked as complete successfully
        print('Event marked as complete successfully!');
        setState(() {
          final event = myEvents.firstWhere((event) => event['event_url'] == eventUrl);
          event['is_complete'] = true;
        });
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Event marked as complete')),
        );
      } else {
        print('Error: ${response.statusCode}, ${response.body}');
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to mark event as complete: ${response.body}')),
        );
      }
    } catch (e) {
      print('Error: $e');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to mark event as complete: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('My Events'),
      ),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : myEvents.isEmpty
              ? Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Text('No events created',
                          style: TextStyle(fontSize: 18)),
                      const SizedBox(height: 20),
                      ElevatedButton(
                        onPressed: () {
                          Navigator.push(
                            context,
                            MaterialPageRoute(
                              builder: (context) => CreateEventScreen(
                                userData: widget.userData,
                              ),
                            ),
                          );
                        },
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.black,
                          foregroundColor: Colors.white,
                        ),
                        child: const Text('Create Event'),
                      ),
                    ],
                  ),
                )
              : ListView.builder(
                  itemCount: myEvents.length,
                  itemBuilder: (context, index) {
                    final event = myEvents[index];
                    final eventUrl = event['event_url']; // Assuming 'event_url' exists
                    return Dismissible(
                      key: Key(eventUrl.toString()),
                      background: Container(
                        color: Colors.red,
                        child: const Icon(Icons.delete, color: Colors.white),
                      ),
                      onDismissed: (direction) {
                        _deleteEvent(eventUrl); // Delete event on swipe
                      },
                      child: Card(
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
                              Text('URL: ${event['event_url']}'),
                              Text('Created At: ${event['created_at']}'),
                              Text('Completed: ${event['is_complete']}'),
                              Text('Username: ${event['username']}'),
                              Row(
                                children: [
                                  IconButton(
                                    icon: Icon(Icons.photo, color: Colors.black),
                                    onPressed: () async {
                                      final result = await Navigator.push(
                                        context,
                                        MaterialPageRoute(
                                          builder: (context) => ImageUploadScreen(
                                            userData: widget.userData,
                                            eventUrl: eventUrl,
                                          ),
                                        ),
                                      );
                                      if (result == true) {
                                        await _markEventAsComplete(eventUrl);
                                      }
                                    },
                                  ),
                                  SizedBox(width: 10),
                                  IconButton(
                                    icon: Icon(Icons.add_to_drive_sharp, color: Colors.black),
                                    onPressed: () {
                                      // Future functionality
                                    },
                                  ),
                                  SizedBox(width: 10),
                                  IconButton(
                                    icon: Icon(Icons.g_mobiledata_outlined, color: Colors.black),
                                    onPressed: () {
                                      // Future functionality
                                    },
                                  ),
                                ],
                              ),
                            ],
                          ),
                          trailing: IconButton(
                            icon: Icon(Icons.delete, color: Colors.red),
                            onPressed: () {
                              _deleteEvent(eventUrl); // Delete event on button press
                            },
                          ),
                        ),
                      ),
                    );
                  },
                ),
    );
  }
}
