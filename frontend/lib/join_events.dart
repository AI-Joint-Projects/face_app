import 'package:flutter/material.dart';
import 'package:qr_code_scanner/qr_code_scanner.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class JoinEventsScreen extends StatefulWidget {
  const JoinEventsScreen({super.key, required this.userData});
  final Map<String, dynamic> userData;

  @override
  _JoinEventsScreenState createState() => _JoinEventsScreenState();
}

class _JoinEventsScreenState extends State<JoinEventsScreen> {
  final _urlController = TextEditingController();
  String? _scannedUrl;
  final GlobalKey qrKey = GlobalKey(debugLabel: 'QR');

  QRViewController? controller;

  @override
  void dispose() {
    _urlController.dispose();
    controller?.dispose();
    super.dispose();
  }

  void _onQRViewCreated(QRViewController controller) {
    this.controller = controller;
    controller.scannedDataStream.listen((scanData) {
      setState(() {
        _scannedUrl = scanData.code;
        _urlController.text = _scannedUrl ?? '';
      });
      controller.pauseCamera();
      Navigator.of(context).pop();
    });
  }

  Future<void> _joinEvent() async {
    final url = _urlController.text.isNotEmpty ? _urlController.text : _scannedUrl;
    if (url == null || url.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter a URL or scan a QR code')),
      );
      return;
    }

    try {
      final response = await http.put(
        Uri.parse('https://x200lnxp-8000.inc1.devtunnels.ms/events/join-event'),
        body: jsonEncode({'url': url}),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer ${widget.userData['access_token']}',
        },
      );

      if (response.statusCode == 202) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Event joined successfully!')),
        );
      } else {
        print('Error: ${response.statusCode} - ${response.body}');
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Failed to join event: ${response.body}')),
        );
      }
    } catch (e) {
      print('Error: $e');
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Failed to join event')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Join Events'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Join Event',
              style: TextStyle(
                fontSize: 24.0,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16.0),
            const Text(
              'Enter URL or Scan QR code:',
              style: TextStyle(fontSize: 16.0),
            ),
            const SizedBox(height: 16.0),
            TextField(
              controller: _urlController,
              decoration: const InputDecoration(
                labelText: 'URL',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 16.0),
            ElevatedButton(
              onPressed: () {
                showModalBottomSheet(
                  context: context,
                  builder: (BuildContext context) {
                    return SizedBox(
                      height: 300,
                      child: QRView(
                        key: qrKey,
                        onQRViewCreated: _onQRViewCreated,
                        overlay: QrScannerOverlayShape(
                          borderColor: Colors.red,
                          borderRadius: 10,
                          borderLength: 30,
                          borderWidth: 10,
                          cutOutSize: 300,
                        ),
                      ),
                    );
                  },
                );
              },
              child: const Text('Scan QR Code'),
            ),
            const SizedBox(height: 16.0),
            if (_scannedUrl != null)
              Text(
                'Scanned URL: $_scannedUrl',
                style: const TextStyle(fontSize: 16.0),
              ),
            const SizedBox(height: 32.0),
            Center(
              child: ElevatedButton(
                onPressed: _joinEvent,
                child: const Text('Join Event'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
