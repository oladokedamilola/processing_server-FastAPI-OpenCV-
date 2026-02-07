# Image Processing Server

A high-performance computer vision server built with FastAPI and OpenCV, specializing in real-time image and video analysis. This production-ready solution provides advanced image processing capabilities for surveillance, monitoring, and multimedia analysis applications.

## 🎯 Core Image Processing Capabilities

### **Detection & Recognition**
- **Multi-class Object Detection**: Identify 80+ object categories using YOLOv8 Nano
- **Human Detection**: Specialized people detection using HOG descriptors and Haar Cascades
- **Vehicle Recognition**: Car, truck, motorcycle, and bus detection
- **Motion Analysis**: Frame differencing for movement detection and activity monitoring
- **Bounding Box Intelligence**: Precise object localization with confidence scoring

### **Image Analysis Features**
- **Real-time Processing**: Sub-3-second response times for image analysis
- **Confidence Thresholding**: Adjustable sensitivity (0.1-0.9) for detection accuracy
- **Size-based Filtering**: Configurable minimum object dimensions for filtering
- **Multi-model Processing**: Ensemble approach combining YOLOv8 with traditional CV algorithms

## 🔧 Processing Functionality

### **Image Processing Pipeline**
```
Input → Validation → Decoding → Preprocessing → Detection → Analysis → Response
    ↓           ↓           ↓           ↓           ↓           ↓           ↓
File Check → Format Verify → OpenCV Load → Optimization → Model Inference → JSON Format → API Return
```

### **Supported Operations**
- **Single Image Analysis**: Immediate processing with detailed detection results
- **Batch Image Processing**: Multiple image analysis in optimized sequences
- **Video Frame Extraction**: Intelligent frame sampling for video analysis
- **Format Conversion**: Automatic normalization across different image formats
- **Metadata Extraction**: Image dimensions, properties, and quality assessment

## 📊 Detection & Output

### **Detection Results Structure**
```json
{
  "detections": [
    {
      "label": "person",
      "confidence": 0.92,
      "bbox": [120, 85, 310, 480],
      "dimensions": {"width": 190, "height": 395},
      "position": {"center_x": 215, "center_y": 282.5}
    },
    {
      "label": "car",
      "confidence": 0.87,
      "bbox": [450, 200, 620, 320],
      "dimensions": {"width": 170, "height": 120},
      "position": {"center_x": 535, "center_y": 260}
    }
  ],
  "image_analysis": {
    "resolution": "1920x1080",
    "color_profile": "RGB",
    "detection_summary": {
      "total_objects": 7,
      "people_count": 3,
      "vehicles_count": 2,
      "other_objects": 2
    },
    "processing_metrics": {
      "inference_time": 1.23,
      "total_processing_time": 2.45,
      "frames_per_second": 40.8
    }
  }
}
```

### **Advanced Analysis Features**
- **Density Estimation**: Object count per region/quadrant
- **Activity Heatmaps**: Movement concentration visualization
- **Object Tracking**: Basic trajectory analysis across video frames
- **Scene Understanding**: Dominant object and activity classification

## 🖼️ Supported Image Formats & Specifications

### **Input Formats**
- **Images**: JPEG, PNG, BMP, TIFF, WebP
- **Videos**: MP4, AVI, MOV, MKV (frame-by-frame processing)
- **Maximum Size**: 10MB per image, 50MB per video
- **Color Spaces**: Automatic conversion to RGB for processing

### **Processing Specifications**
- **Resolution Handling**: Automatic scaling for optimal processing
- **Aspect Ratio Preservation**: Maintains original image proportions
- **Color Normalization**: Standardized color processing pipeline
- **Noise Reduction**: Pre-processing filters for improved detection

## ⚡ Performance & Optimization

### **Speed & Efficiency**
- **Image Processing**: < 3 seconds for standard 1080p images
- **Model Inference**: Optimized YOLOv8 Nano with 6.2MB footprint
- **Memory Management**: Efficient processing within 512MB constraints
- **Parallel Processing**: Concurrent image analysis capabilities

### **Quality & Accuracy**
- **Detection Rate**: >85% accuracy for person detection
- **Precision Control**: Adjustable confidence thresholds
- **False Positive Reduction**: Size-based filtering and multi-model validation
- **Lighting Adaptation**: Robust performance across varied lighting conditions

## 🔌 API Endpoints for Image Processing

### **Primary Processing Endpoints**
- **`POST /api/v1/process/image`** - Single image analysis
- **`POST /api/v1/process/batch-images`** - Multiple image processing
- **`POST /api/v1/process/video`** - Video frame analysis
- **`POST /api/v1/analyze/advanced`** - Enhanced analysis with heatmaps

### **Configuration & Control**
- **`POST /api/v1/config/threshold`** - Adjust detection sensitivity
- **`GET /api/v1/models/status`** - Check model availability and versions
- **`POST /api/v1/processing/mode`** - Switch between speed/accuracy modes

## 🛠️ Technical Implementation

### **Processing Stack**
- **Primary Engine**: OpenCV with optimized image operations
- **Detection Core**: YOLOv8 Nano for real-time object detection
- **Supplementary Models**: HOG + SVM for people, Haar Cascades for faces
- **Motion Analysis**: Background subtraction and frame differencing

### **Optimization Features**
- **Model Caching**: On-demand loading with intelligent retention
- **Image Buffering**: Efficient memory management for batch processing
- **Pre-processing Pipeline**: Automated optimization for detection quality
- **Result Caching**: Temporary storage for repeated analysis requests

## 📈 Use Cases & Applications

### **Surveillance & Security**
- Real-time person and vehicle detection
- Intrusion detection and perimeter monitoring
- Crowd density analysis and people counting
- Suspicious activity identification

### **Media & Content Analysis**
- Image content classification and tagging
- Object recognition for media libraries
- Video summarization through key frame extraction
- Automated content moderation

### **Industrial Applications**
- Quality control through object detection
- Inventory counting and management
- Safety compliance monitoring
- Process automation via visual recognition

## 🔍 Advanced Features

### **Custom Detection Configurations**
- Class-specific sensitivity settings
- Region-of-interest based analysis
- Time-of-day adaptive processing
- Environment-aware detection parameters

### **Analysis Enhancements**
- Motion vector calculation
- Object size estimation and measurement
- Color-based object classification
- Pattern recognition within detected objects

### **Output Customization**
- Multiple coordinate formats (pixels, percentages, normalized)
- Custom attribute extraction
- Format-specific output (JSON, XML, CSV)
- Webhook integration for automated workflows


**Status**: Production Ready | **Image Processing Focus**: Core Functionality  
**Optimized For**: Real-time analysis, High-volume processing, Accurate detection