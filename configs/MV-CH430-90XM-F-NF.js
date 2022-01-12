var grabber = grabbers[0];
//Interface
grabber.InterfacePort.set("LineSelector", "TTLIO11");
grabber.InterfacePort.set("LineMode", "Input");    
grabber.InterfacePort.set("LineInputToolSelector", "LIN1"); 
grabber.InterfacePort.set("LineInputToolSource", "TTLIO11"); 
grabber.InterfacePort.set("LineInputToolActivation", "RisingEdge");
//Remote
grabber.RemotePort.set("TriggerMode", "On");
grabber.RemotePort.set("TriggerSource", "LinkTrigger0");
grabber.RemotePort.set("Width", 5120);
grabber.RemotePort.set("Height", 5120);
//Device
grabber.DevicePort.set("CameraControlMethod", "RC");
grabber.DevicePort.set("CycleTriggerSource", "LIN1");
grabber.DevicePort.set("ExposureTime",100);