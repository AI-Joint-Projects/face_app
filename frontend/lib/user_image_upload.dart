import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

class ImageUploadScreen extends StatefulWidget {
  const ImageUploadScreen({super.key, required this.userData, required this.eventUrl});
  final Map<String, dynamic> userData;
  final String eventUrl;

  @override
  _ImageUploadScreenState createState() => _ImageUploadScreenState();
}

class _ImageUploadScreenState extends State<ImageUploadScreen> {
  final ImagePicker _picker = ImagePicker();
  List<XFile>? _images = [];

  Future<void> _selectImages() async {
    final List<XFile>? selectedImages = await _picker.pickMultiImage();
    if (selectedImages != null) {
      setState(() {
        _images = selectedImages;
      });
    }
  }

  Future<void> _uploadImages() async {
    if (_images == null || _images!.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('No images selected')),
      );
      return;
    }

    // Assuming event_url is passed within userData
    final event = widget.eventUrl;
    final url = Uri.parse(
        'https://x200lnxp-8000.inc1.devtunnels.ms/upload-images/local-storage?event=$event');

    var request = http.MultipartRequest('PUT', url);
    request.headers['Authorization'] = 'Bearer ${widget.userData['access_token']}';

    for (var image in _images!) {
      print('Adding image: ${image.path}');
      request.files.add(await http.MultipartFile.fromPath('files', image.path));
    }

    try {
      final response = await request.send();
      if (response.statusCode == 202) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Images uploaded successfully')),
        );
        setState(() {
          _images = [];
        });
        Navigator.pop(context, true); // Return true to indicate success
      } else {
        print('Error: ${response.statusCode}');
        print(await response.stream.bytesToString());
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to upload images')),
        );
        Navigator.pop(context, false); // Return false to indicate failure
      }
    } catch (e) {
      print('Error: $e');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to upload images: $e')),
      );
      Navigator.pop(context, false); // Return false to indicate failure
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Upload Images'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: _selectImages,
              child: Text('Select Images'),
            ),
            if (_images != null && _images!.isNotEmpty)
              ElevatedButton(
                onPressed: _uploadImages,
                child: Text('Upload Images'),
              ),
          ],
        ),
      ),
    );
  }
}
