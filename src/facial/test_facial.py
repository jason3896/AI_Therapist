from facial_emotion import FacialEmotionAnalyzer
import cv2

def test_single_image():
    # Load an image
    img_path = input("Enter the path to an image file: ")
    image = cv2.imread(img_path)
    
    if image is None:
        print(f"Error: Could not load image at {img_path}")
        return
    
    # Initialize analyzer
    analyzer = FacialEmotionAnalyzer()
    
    # Analyze emotion
    result = analyzer.analyze_emotion(image)
    
    # Print results
    if result["success"]:
        print(f"Detected emotion: {result['emotion']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("All emotions:")
        for emotion, score in result.get("all_emotions", {}).items():
            print(f"  {emotion}: {score:.2f}%")
    else:
        print("No face detected or error in analysis.")

if __name__ == "__main__":
    test_single_image()