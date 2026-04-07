import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:pytorch_lite/pytorch_lite.dart';
import 'package:flutter/services.dart'; 
import 'dart:io';

late List<CameraDescription> _cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  // Инициализируем камеры телефона
  _cameras = await availableCameras();
  runApp(const AgroScanApp());
}

class AgroScanApp extends StatelessWidget {
  const AgroScanApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(), // Темная тема для "про" вида
      home: const CameraScreen(),
    );
  }
}

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  late CameraController controller;
  ClassificationModel? _model;
  
  // список растений и выбранное по умолчанию
  String selectedCrop = "Tomato";
  final List<String> crops = ["Tomato", "Apple", "Grape", "Peach", "Potato", "Strawberry"];
  
  String _result = "Выберите растение и наведите на лист";
  bool _isProcessing = false;

  @override
  void initState() {
    super.initState();
    // Запускаем камеру 
    controller = CameraController(_cameras[0], ResolutionPreset.medium);
    controller.initialize().then((_) {
      if (!mounted) return;
      setState(() {});
      // Загружаем начальную модель
      loadModel(selectedCrop); 
    });
  }

  // функция загрузки модели
  Future loadModel(String cropName) async {
    try {
      setState(() => _result = "Загрузка модели $cropName...");
      
      String labelPath = "assets/models/${cropName}_labels.txt";
      
      // 1. Читаем текстовый файл с лейблами
      String labelsData = await rootBundle.loadString(labelPath);
      
      // считаем количество колво болезней
      int classCount = labelsData.split('\n').where((line) => line.trim().isNotEmpty).length;
      
      // передаем это число в модель
      _model = await PytorchLite.loadClassificationModel(
        "assets/models/$cropName.ptl", 
        224, // Ширина
        224, // Высота
        classCount, 
        labelPath: labelPath,
      );
      
      setState(() => _result = "Модель $cropName готова (классов: $classCount)");
      print("Модель $cropName успешно загружена! Классов: $classCount");
    } catch (e) {
      setState(() => _result = "Ошибка загрузки: $e");
      print("Ошибка загрузки модели: $e");
    }
  }

  // Логика анализа
  Future runInference() async {
    if (_model == null || _isProcessing) return;
    
    setState(() => _isProcessing = true);
    
    try {
      // делаем снимок
      final image = await controller.takePicture();
      
      // прогоняем через нейросеть
      String prediction = await _model!.getImagePrediction(
        await File(image.path).readAsBytes(),
      );

      setState(() {
        _result = "Результат: $prediction";
      });
    } catch (e) {
      setState(() => _result = "Ошибка анализа: $e");
    } finally {
      setState(() => _isProcessing = false);
    }
  }

  @override
  void dispose() {
    controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!controller.value.isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }
    
    return Scaffold(
      appBar: AppBar(
        title: const Text("AgroScan AI"),
        centerTitle: true,
      ),
      body: Column(
        children: [
          // Выпадающее меню для выбора культуры
          Padding(
            padding: const EdgeInsets.all(10.0),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12),
              decoration: BoxDecoration(
                color: Colors.grey[850],
                borderRadius: BorderRadius.circular(10),
              ),
              child: DropdownButton<String>(
                value: selectedCrop,
                isExpanded: true,
                underline: const SizedBox(),
                items: crops.map((String crop) {
                  return DropdownMenuItem<String>(
                    value: crop,
                    child: Text(crop),
                  );
                }).toList(),
                onChanged: (newValue) {
                  if (newValue != null) {
                    setState(() {
                      selectedCrop = newValue;
                    });
                    loadModel(newValue);
                  }
                },
              ),
            ),
          ),
          
          // Вид с камеры
          Expanded(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 10),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(15),
                child: CameraPreview(controller),
              ),
            ),
          ),
          
          // Результат
          Padding(
            padding: const EdgeInsets.all(20.0),
            child: Text(
              _result, 
              textAlign: TextAlign.center,
              style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)
            ),
          ),
          
          // Кнопка анализа
          Padding(
            padding: const EdgeInsets.only(bottom: 30),
            child: SizedBox(
              width: 200,
              height: 50,
              child: ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green[700],
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(25))
                ),
                onPressed: runInference,
                child: _isProcessing 
                  ? const CircularProgressIndicator(color: Colors.white) 
                  : const Text("АНАЛИЗ", style: TextStyle(fontSize: 16, color: Colors.white)),
              ),
            ),
          ),
        ],
      ),
    );
  }
}