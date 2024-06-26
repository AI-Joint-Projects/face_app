import 'dart:io';
import 'package:flutter/material.dart';
import 'package:qr_flutter/qr_flutter.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:path_provider/path_provider.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:image_gallery_saver/image_gallery_saver.dart';

class CreateEventScreen extends StatefulWidget {
  const CreateEventScreen({super.key, required this.userData});
  final Map<String, dynamic> userData;

  @override
  _CreateEventScreenState createState() => _CreateEventScreenState();
}

class _CreateEventScreenState extends State<CreateEventScreen> {
  final TextEditingController _titleController = TextEditingController();
  String? _eventLink;
  bool _isLoading = false;

  @override
  void dispose() {
    _titleController.dispose();
    super.dispose();
  }

  Future<void> _createEvent() async {
    setState(() {
      _isLoading = true;
    });

    final title = _titleController.text;

    final url = Uri.parse(
        'https://x200lnxp-8000.inc1.devtunnels.ms/events/create-event'); // Replace with your API endpoint

    try {
      final response = await http.post(url,
          body: jsonEncode({
            'event_name': title,
            "_id": "string",
            "event_url": "string",
            "created_at": "2024-05-28T15:36:51.715008",
            "images_directory": "string",
            "is_complete": false,
            "username": "string"
            
          }),
          headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ${widget.userData['access_token']}'
          });

      setState(() {
        _isLoading = false;
      });

      if (response.statusCode == 201) {
        final data = jsonDecode(response.body);
        setState(() {
          _eventLink = data['event_link'];
        });
      } else {
        print('Error: ${response.statusCode}');
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to create event')),
        );
      }
    } catch (e) {
      setState(() {
        _isLoading = false;
      });
      print('Error: $e');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to create event')),
      );
    }
  }

  Future<void> _saveQrCode() async {
    if (_eventLink == null) return;

    // Request storage permission
    final status = await Permission.storage.request();
    if (status.isGranted) {
      try {
        // Get temporary directory
        final directory = await getTemporaryDirectory();
        final path = '${directory.path}/qr_code.png';

        // Validate the QR code data
        final qrValidationResult = QrValidator.validate(
          data: _eventLink!,
          version: QrVersions.auto,
          errorCorrectionLevel: QrErrorCorrectLevel.L,
        );

        if (qrValidationResult.status == QrValidationStatus.valid) {
          final qrCode = qrValidationResult.qrCode!;
          final painter = QrPainter.withQr(
            qr: qrCode,
            color: Color.fromARGB(255, 0, 0, 0),
            gapless: true,
            emptyColor: Colors.white,
          );

          // Increase the resolution here
          final picData = await painter.toImageData(1000); // Set a higher resolution
          final bytes = picData!.buffer.asUint8List();

          final file = File(path);
          await file.writeAsBytes(bytes);

          // Save the image to the gallery
          await ImageGallerySaver.saveFile(path);
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('QR code saved to gallery!')),
          );
        }
      } catch (e) {
        print('Error saving QR code: $e');
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to save QR code')),
        );
      }
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Storage permission denied.')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Create Event'),
      ),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Title',
              style: TextStyle(
                fontSize: 18.0,
                fontWeight: FontWeight.bold,
              ),
            ),
            Container(
              decoration: BoxDecoration(
              border: Border.all(
                color: Colors.grey,
                width: 1.0,
              ),
              borderRadius: BorderRadius.circular(4.0),
              ),
              child: TextField(
              controller: _titleController,
              decoration: InputDecoration(
                hintText: 'Enter event title',
                border: InputBorder.none,
                contentPadding: EdgeInsets.all(16.0),
              ),
              ),
            ),
            SizedBox(height: 32.0),
            if (_isLoading)
              Center(child: CircularProgressIndicator())
            else if (_eventLink != null)
              Column(
                children: [
                  QrImageView(
                    data: _eventLink!,
                    version: QrVersions.auto,
                    size: 200.0,
                  ),
                  SizedBox(height: 16.0),
                  ElevatedButton(
                    onPressed: _saveQrCode,
                    child: Text('Download QR Code'),
                  ),
                ],
              )
            else
              Center(
  child: ElevatedButton(
    onPressed: _createEvent,
    style: ElevatedButton.styleFrom(
      backgroundColor: Colors.black, // Background color
    ),
    child: Text(
      'Create Event',
      style: TextStyle(
        color: Colors.white, // Text color
      ),
    ),
  ),
),

          ],
        ),
      ),
    );
  }
}
