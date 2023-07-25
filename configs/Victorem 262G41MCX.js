var grabber = grabbers[0];
//Interface
grabber.InterfacePort.set("LineSelector", "TTLIO11");
grabber.InterfacePort.set("LineMode", "Input");    
grabber.InterfacePort.set("LineInputToolSelector", "LIN1"); 
grabber.InterfacePort.set("LineInputToolSource", "TTLIO11"); 
grabber.InterfacePort.set("LineInputToolActivation", "RisingEdge");
//Remote
grabber.RemotePort.set("ExposureMode", "Edge_Triggered_Programmable");
grabber.RemotePort.set("TriggerSource", "CoaXPress_Trigger_Input");
grabber.RemotePort.set("TriggerEdgeLevel", "Rising_Edge_High_Level");
grabber.RemotePort.set("Width", 5120);  // ATTENTION: setting this option may cause camera freezing
grabber.RemotePort.set("Height", 5120);  // ATTENTION: setting this option may cause camera freezing
//Device
grabber.DevicePort.set("CameraControlMethod", "RC");
grabber.DevicePort.set("CycleTriggerSource", "LIN1");
grabber.DevicePort.set("ExposureTime",100);