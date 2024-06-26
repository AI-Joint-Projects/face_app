import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:archive/archive.dart';
import 'dart:typed_data';

class EventImagesScreen extends StatefulWidget {
  const EventImagesScreen({Key? key, required this.userData, required this.eventUrl}) : super(key: key);
  final String eventUrl;
  final Map<String, dynamic> userData;

  @override
  _EventImagesScreenState createState() => _EventImagesScreenState();
}

class _EventImagesScreenState extends State<EventImagesScreen> {
  List<Uint8List> imageBytes = [];
  bool isLoading = true;
  String? errorMessage;

  @override
  void initState() {
    super.initState();
    _fetchEventImages();
  }

  Future<void> _fetchEventImages() async {
    try {
      final response = await http.get(
        Uri.parse('https://x200lnxp-8000.inc1.devtunnels.ms/events/event-images')
            .replace(queryParameters: {"event_url": widget.eventUrl}),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer ${widget.userData['access_token']}',
        },
      );

      if (response.statusCode == 200) {
        final bytes = response.bodyBytes;
        final archive = ZipDecoder().decodeBytes(bytes);
        final List<Uint8List> extractedImages = [];

        for (var file in archive.files) {
          if (file.isFile) {
            extractedImages.add(file.content as Uint8List);
          }
        }

        setState(() {
          imageBytes = extractedImages;
          isLoading = false;
        });
      } else {
        setState(() {
          errorMessage = 'Failed to load images: ${response.statusCode}';
          isLoading = false;
        });
        print('Error: ${response.statusCode}, ${response.body}');
      }
    } catch (e) {
      setState(() {
        errorMessage = 'Failed to load images: $e';
        isLoading = false;
      });
      print('Error: $e');
    }
  }

  void _showImageDialog(Uint8List imageBytes) {
    showDialog(
      context: context,
      builder: (context) => Dialog(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Expanded(
              child: InteractiveViewer(
                child: Image.memory(imageBytes),
              ),
            ),
            TextButton(
              onPressed: () {
                Navigator.pop(context);
              },
              child: const Text('Close'),
            ),
          ],
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Event Images'),
      ),
      body: isLoading
          ? const Center(child: CircularProgressIndicator())
          : errorMessage != null
              ? Center(child: Text(errorMessage!))
              : Padding(
                  padding: const EdgeInsets.all(8.0), // Add padding to the GridView
                  child: GridView.builder(
                    gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                      crossAxisCount: 4,
                      childAspectRatio: 1,
                      crossAxisSpacing: 8.0, // Space between columns
                      mainAxisSpacing: 8.0, // Space between rows
                    ),
                    itemCount: imageBytes.length,
                    itemBuilder: (context, index) {
                      final image = imageBytes[index];
                      return GestureDetector(
                        onTap: () => _showImageDialog(image),
                        child: Image.memory(image, fit: BoxFit.cover),
                      );
                    },
                  ),
                ),
    );
  }
}
