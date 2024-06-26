import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:reaco/dashboard.dart';
import 'package:image_picker/image_picker.dart';

class UploadScreen extends StatefulWidget {
  const UploadScreen({super.key, required this.userData});

  final Map<String, dynamic> userData;

  @override
  _UploadScreenState createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> {
  late VideoPlayerController _videoController;
  File? _videoFile;
  bool _isRecording = false;
  Duration _recordedDuration = Duration.zero;
  final _formKey = GlobalKey<FormState>();
  bool _isLoading = false; // Add a loading state

  @override
  void initState() {
    super.initState();
    _videoController = VideoPlayerController.file(File(''))
      ..initialize().then((_) {
        setState(() {});
      });
  }

  @override
  void dispose() {
    _videoController.dispose();
    super.dispose();
  }

  Future<void> _startRecording() async {
    if (_isRecording) return;

    final pickedFile = await ImagePicker().pickVideo(
      source: ImageSource.camera,
      maxDuration: const Duration(seconds: 5),
    );
    if (pickedFile != null) {
      setState(() {
        _videoFile = File(pickedFile.path);
      });

      _videoController = VideoPlayerController.file(_videoFile!)
        ..initialize().then((_) {
          setState(() {
            _recordedDuration = _videoController.value.duration;
            _isRecording = false;
          });
        });
    } else {
      // User canceled the recording
      // Handle the scenario here
    }
  }

  Future<void> _uploadVideo() async {
    if (_formKey.currentState!.validate()) {
      if (_videoFile == null) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Please record a video')),
        );
        return;
      }

      // Assuming your backend API endpoint is 'http://your-backend-url/take-images'
      final apiUrl = 'https://x200lnxp-8000.inc1.devtunnels.ms/users/process-video';

      setState(() {
        _isLoading = true; // Set loading state to true
      });

      try {
        // Get the auth token from the provided userData
        final authToken = widget.userData['access_token'];

        final request = http.MultipartRequest('POST', Uri.parse(apiUrl));

        // Add authorization headers
        request.headers['Authorization'] = 'Bearer $authToken';

        final multipartFile = http.MultipartFile.fromBytes(
          'video',
          _videoFile!.readAsBytesSync(),
          filename: _videoFile!.path.split('/').last,
        );
        request.files.add(multipartFile);

        final response = await request.send();

        if (response.statusCode == 202) {
          // Successful upload, navigate to the next screen
          Navigator.push(
            context,
            MaterialPageRoute(builder: (context) => DashBoard(userData: widget.userData)),
          );
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Video Uploaded Successfully')),
          );
        } else if (response.statusCode == 400) {
          // Handle 400 error (Face not found)
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Face is not found')),
          );
        } else {
          // Handle other errors
          print('Error uploading video: ${response.statusCode}');
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Error uploading video: ${response.statusCode}')),
          );
        }
      } catch (e) {
        // Handle network errors
        print('Error: $e');
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error uploading video: $e')),
        );
      } finally {
        setState(() {
          _isLoading = false; // Reset loading state to false
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Upload'),
      ),
      body: Form(
        key: _formKey,
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Text(
                'RECORD VIDEO',
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  fontSize: 20,
                ),
              ),
              const SizedBox(height: 10),
              const Text(
                'Record a video with one face for 5 seconds',
                style: TextStyle(fontSize: 16),
              ),
              const SizedBox(height: 20),
              if (_videoFile != null)
                Stack(
                  alignment: Alignment.center,
                  children: [
                    SizedBox(
                      width: 250,
                      height:400,
                      child: _videoController.value.isInitialized
                          ? AspectRatio(
                              aspectRatio: 16 / 9,
                              child: VideoPlayer(_videoController),
                            )
                          : const CircularProgressIndicator(),
                    ),
                    IconButton(
                      onPressed: () {
                        setState(() {
                          _videoController.value.isPlaying
                              ? _videoController.pause()
                              : _videoController.play();
                        });
                      },
                      icon: Icon(_videoController.value.isPlaying
                          ? Icons.pause_circle
                          : Icons.play_circle),
                      color: Colors.white,
                      iconSize: 40,
                    ),
                  ],
                )
              else
                SizedBox(
                  width: 200,
                  height: 200,
                  child: ElevatedButton(
                    onPressed: _startRecording,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.grey[300],
                      padding: EdgeInsets.zero,
                    ),
                    child: Icon(
                      _isRecording ? Icons.stop : Icons.videocam,
                      color: Colors.black,
                      size: 80,
                    ),
                  ),
                ),
              const SizedBox(height: 20),
              if (_recordedDuration > const Duration(seconds: 7))
                Text(
                  'Video is too long. Please record a video under 5 seconds.',
                  style: TextStyle(color: Colors.red),
                ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: _videoFile != null && _recordedDuration <= const Duration(seconds: 7)
                    ? _uploadVideo
                    : null,
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(horizontal: 40),
                  backgroundColor: Colors.black,
                ),
                child: const Text(
                  'NEXT',
                  style: TextStyle(color: Colors.white),
                ),
              ),
              const SizedBox(height: 20),
              if (_videoFile != null)
                ElevatedButton(
                  onPressed: () {
                    setState(() {
                      _videoFile = null;
                      _recordedDuration = Duration.zero;
                      _videoController.dispose();
                      _videoController = VideoPlayerController.file(File(''))
                        ..initialize().then((_) {
                          setState(() {});
                        });
                    });
                  },
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(horizontal: 40),
                    backgroundColor: Colors.grey[300],
                  ),
                  child: const Text(
                    'Record Again',
                    style: TextStyle(color: Colors.black),
                  ),
                ),
              // Add a progress bar for loading
              if (_isLoading)
                const LinearProgressIndicator(
                  color: Colors.black,
                ),
              const SizedBox(height: 20),
            ],
          ),
        ),
      ),
    );
  }
}