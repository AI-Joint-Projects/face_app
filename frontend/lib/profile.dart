import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:archive/archive.dart';
import 'package:reaco/login.dart';
import 'package:reaco/upload_images.dart';

class ProfileScreen extends StatefulWidget {
  final Map<String, dynamic> userData;

  const ProfileScreen({Key? key, required this.userData}) : super(key: key);

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  File? _image;
  String? _profilePictureUrl;
  String? _username;
  int? _eventsJoinedCount;
  int? _eventsCreatedCount;
  List<Uint8List>? _faceImages;

  final ImagePicker _picker = ImagePicker();


  Future<void> _getImage() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      setState(() {
        _image = File(image.path);
      });
      await _uploadProfilePicture();
    }
  }

  Future<void> _uploadProfilePicture() async {
    if (_image != null) {
      final authToken = widget.userData['access_token'];
      final uploadUrl = Uri.parse('https://x200lnxp-8000.inc1.devtunnels.ms/profile/upload-profile-picture');

      final request = http.MultipartRequest('PUT', uploadUrl);
      request.headers['Authorization'] = 'Bearer $authToken';
      request.files.add(await http.MultipartFile.fromPath('image', _image!.path));

      final response = await request.send();

      if (response.statusCode == 202) {
        final responseBody = await response.stream.bytesToString();
        final data = jsonDecode(responseBody);
        setState(() {
          _profilePictureUrl = data['profile_picture'];
        });
        _getProfilePicture(); // To refresh the profile picture from the server
      } else {
        print('Error uploading image: ${response.statusCode}');
      }
    }
  }

  Future<void> _getProfilePicture() async {
    final authToken = widget.userData['access_token'];
    final getUrl = Uri.parse('https://x200lnxp-8000.inc1.devtunnels.ms/profile/profile-picture');
    print('Fetching profile picture from: $getUrl'); // Add this line
    final response = await http.get(getUrl, headers: {
      'Authorization': 'Bearer $authToken',
    });

    if (response.statusCode == 200) {
      setState(() {
        _profilePictureUrl = response.body;
      });
    } else {
      print('Error retrieving image: ${response.statusCode}');
    }
  }

  Future<void> _fetchUsername() async {
    final authToken = widget.userData['access_token'];
    final getUrl = Uri.parse('https://x200lnxp-8000.inc1.devtunnels.ms/profile/get-username');

    final response = await http.get(getUrl, headers: {
      'Authorization': 'Bearer $authToken',
    });

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      setState(() {
        _username = data["username"];
      });
    } else {
      print('Error fetching username: ${response.statusCode}');
    }
  }

  Future<void> _fetchEventsJoinedCount() async {
    final authToken = widget.userData['access_token'];
    final getUrl = Uri.parse('https://x200lnxp-8000.inc1.devtunnels.ms/profile/joined-count');

    final response = await http.get(getUrl, headers: {
      'Authorization': 'Bearer $authToken',
    });

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      setState(() {
        _eventsJoinedCount = data['joined_events_count'];
      });
    } else {
      print('Error fetching events joined count: ${response.statusCode}');
    }
  }

  Future<void> _fetchEventsCreatedCount() async {
    final authToken = widget.userData['access_token'];
    final getUrl = Uri.parse('https://x200lnxp-8000.inc1.devtunnels.ms/profile/created-count');

    final response = await http.get(getUrl, headers: {
      'Authorization': 'Bearer $authToken',
    });

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      setState(() {
        _username = data['username'];
        _eventsCreatedCount = data['created_events_count'];
      });
    } else {
      print('Error fetching events created count: ${response.statusCode}');
    }
  }

  Future<void> _fetchFaceImages() async {
    final authToken = widget.userData['access_token'];
    final getUrl = Uri.parse('https://x200lnxp-8000.inc1.devtunnels.ms/profile/get-faces');

    final response = await http.get(getUrl, headers: {
      'Authorization': 'Bearer $authToken',
    });

    if (response.statusCode == 200) {
      final zipBytes = response.bodyBytes;
      List<Uint8List> imageList = [];
      final archive = ZipDecoder().decodeBytes(zipBytes);

      for (final file in archive) {
        if (file.isFile) {
          imageList.add(Uint8List.fromList(file.content));
        }
      }

      setState(() {
        _faceImages = imageList;
      });
    } else {
      print('Error fetching face images: ${response.statusCode}');
    }
  }

  Future<void> _deleteUser() async {
    final authToken = widget.userData['access_token'];
    final deleteUrl = Uri.parse('https://x200lnxp-8000.inc1.devtunnels.ms/profile/delete-user');

    final response = await http.delete(deleteUrl, headers: {
      'Authorization': 'Bearer $authToken',
    });

    if (response.statusCode == 200) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('User deleted successfully')),
      );
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => SCREENN()),
      );
    } else {
      print('Error deleting user: ${response.statusCode}');
    }
  }

  @override
  void initState() {
    super.initState();
    _getProfilePicture();
    _fetchUsername();
    _fetchEventsJoinedCount();
    _fetchEventsCreatedCount();
    _fetchFaceImages(); // Fetch face images on init
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Profile'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            GestureDetector(
              onTap: _getImage,
              child: CircleAvatar(
                radius: 40,
                backgroundImage: _profilePictureUrl != null
                    ? NetworkImage(_profilePictureUrl!)
                    : null,
                child: _profilePictureUrl == null
                    ? const Icon(Icons.person, size: 40)
                    : null,
              ),
            ),
            const SizedBox(height: 20),
            if (_username != null)
              Text(
                'Username: $_username',
                style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
              ),
            const SizedBox(height: 10),
            if (_eventsJoinedCount != null)
              Text(
                'Events Joined: $_eventsJoinedCount',
                style: const TextStyle(fontSize: 16),
              ),
            const SizedBox(height: 10),
            if (_eventsCreatedCount != null)
              Text(
                'Events Created: $_eventsCreatedCount',
                style: const TextStyle(fontSize: 16),
              ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _getImage,
              child: const Text('Upload Profile Picture'),
            ),
            const SizedBox(height: 20),
            if (_faceImages != null)
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: _faceImages!.map((imageData) {
                  return Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: Image.memory(
                      imageData,
                      width: 100,
                      height: 100,
                    ),
                  );
                }).toList(),
              ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _deleteUser,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.red,
              ),
              child: const Text(
                'Delete User',
                style: TextStyle(color: Colors.white),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class Splash extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Text('Splash Screen'),
      ),
    );
  }
}
