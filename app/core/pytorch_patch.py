"""
Patch for PyTorch 2.6 compatibility with YOLO models
"""
import torch
import sys

def patch_pytorch_for_yolo():
    """Add all necessary safe globals for YOLO model loading"""
    print("Applying PyTorch 2.6 YOLO compatibility patch...")
    
    try:
        # First, add common torch modules
        torch_classes = [
            torch.nn.modules.container.Sequential,
            torch.nn.modules.conv.Conv2d,
            torch.nn.modules.batchnorm.BatchNorm2d,
            torch.nn.modules.activation.SiLU,
            torch.nn.modules.activation.ReLU,
            torch.nn.modules.pooling.MaxPool2d,
            torch.nn.modules.linear.Linear,
            torch.nn.modules.dropout.Dropout,
        ]
        
        # Try to add ultralytics modules
        try:
            from ultralytics.nn.tasks import DetectionModel
            from ultralytics.nn.modules import Conv, Bottleneck, C2f, SPPF, Concat, Detect
            
            ultralytics_classes = [
                DetectionModel,
                Conv, Bottleneck, C2f, SPPF, Concat, Detect
            ]
            
            torch_classes.extend(ultralytics_classes)
            print(f"Added {len(ultralytics_classes)} Ultralytics classes to safe globals")
            
        except ImportError as e:
            print(f"Warning: Could not import Ultralytics modules: {e}")
            print("Trying alternative import approach...")
            
            # Try dynamic import
            try:
                import importlib
                # Import common ultralytics modules
                for module_name in ['Conv', 'Bottleneck', 'C2f', 'SPPF', 'Concat', 'Detect']:
                    try:
                        module = importlib.import_module('ultralytics.nn.modules')
                        if hasattr(module, module_name):
                            torch_classes.append(getattr(module, module_name))
                            print(f"Added ultralytics.nn.modules.{module_name}")
                    except:
                        pass
                
                # Import DetectionModel
                try:
                    tasks_module = importlib.import_module('ultralytics.nn.tasks')
                    if hasattr(tasks_module, 'DetectionModel'):
                        torch_classes.append(tasks_module.DetectionModel)
                        print("Added ultralytics.nn.tasks.DetectionModel")
                except:
                    pass
                    
            except Exception as dynamic_import_error:
                print(f"Dynamic import failed: {dynamic_import_error}")
        
        # Add all classes to safe globals
        torch.serialization.add_safe_globals(torch_classes)
        print(f"PyTorch 2.6 YOLO compatibility patch applied: {len(torch_classes)} classes added")
        
        return True
        
    except Exception as e:
        print(f"Error applying PyTorch patch: {e}", file=sys.stderr)
        
        # Last resort: disable weights_only globally (not recommended for production)
        print("WARNING: Applying workaround - loading models without weights_only=True")
        
        # Monkey-patch torch.load to use weights_only=False
        original_load = torch.load
        
        def patched_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        
        torch.load = patched_load
        print("Applied torch.load patch (weights_only=False)")
        
        return False

# Apply the patch immediately when imported
patch_pytorch_for_yolo()