'''
*****************************************************************************************
*
*        =================================================
*             Pharma Bot Theme (eYRC 2022-23)
*        =================================================
*                                                         
*  This script is intended for implementation of Task 2B   
*  of Pharma Bot (PB) Theme (eYRC 2022-23).
*
*  Filename:			task_2b.py
*  Created:				
*  Last Modified:		8/10/2022
*  Author:				e-Yantra Team
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			[ Team-ID ]
# Author List:		[ Names of team members worked on this file separated by Comma: Name1, Name2, ... ]
# Filename:			task_2b.py
# Functions:		control_logic, read_qr_code
# 					[ Comma separated list of functions in this file ]
# Global variables:	
# 					[ List of global variables defined in this file ]

####################### IMPORT MODULES #######################
## You are not allowed to make any changes in this section. ##
##############################################################
import  sys
import traceback
import time
import os
import math
from zmqRemoteApi import RemoteAPIClient
import zmq
import numpy as np
import cv2
import random
from pyzbar.pyzbar import decode
##############################################################

################# ADD UTILITY FUNCTIONS HERE #################
i=0
lower_range = {'Green':np.array([50,100,100]),
               'Orange':np.array([10,100,20]),
               'Pink':np.array([129,100,100]),
               'Skyblue':np.array([90,50,70]),
               'Blue': np.array([-10,100,100]),
               'Red': np.array([0,50,50]),
               'Black': np.array([0,0,0]),
               'White': np.array([0, 0, 231]),
               'Yellow': np.array([20, 100, 100]),
               'Orange': np.array([10, 50, 70]),
               'Gray': np.array([0, 0, 40])}

upper_range = {'Green':np.array([70,255,255]),
               'Orange':np.array([25,255,255]),
               'Pink':np.array([249,255,255]),
               'Skyblue':np.array([128,255,255]),
               'Blue': np.array([[10,255,255]]),
               'Red': np.array([10,255,255]),
               'Black': np.array([179,100,130]),
               'White': np.array([180, 18, 255]),
               'Yellow': np.array([30, 255, 255]),
               'Orange': np.array([24, 255, 255]),
               'Gray': np.array([180, 18, 230])}

checkpoints = {'A':[1,'L'],
			   'B':[1,'R'],
			   'C':[1,'L'],
			   'D':[1,'R'],
			   'E':[1,'Y'],
			   'F':[1,'R'],
			   'G':[1,'L'],
			   'H':[1,'R'],
			   'I':[1,'Y'],
			   'J':[1,'R'],
			   'K':[1,'L'],
			   'L':[1,'R'],
			   'M':[1,'Y'],
			   'N':[1,'R'],
			   'O':[1,'L'],
			   'P':[1,'R'],
			   'S':[0]}

shapes_qr = {'Pink Cuboid':'package_3','Orange Cone':'package_1','Blue Cylinder':'package_2'}


def sharpen_img(img):
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(img, -1, sharpen_kernel)
    return sharpen



def check_Color1(img_cropped,color):
	hsv_img =cv2.cvtColor(img_cropped, cv2.COLOR_RGB2HSV)
	mask = cv2.inRange(hsv_img, lower_range[color],upper_range[color])
	output = cv2.bitwise_and(img_cropped,img_cropped,mask = mask)
    # show_image(mask)
	isColor = np.sum(mask)
	return isColor
    # print(isColor)
    # if isColor > 0:
    #     return True
    # else:
    #     return False

def check_Color2(img_cropped,color):
    hsv_img =cv2.cvtColor(img_cropped, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_img, lower_range[color],upper_range[color])
    output = cv2.bitwise_and(img_cropped,img_cropped,mask = mask)
    return output


# Used for stopping the bot when yellow part deteted in road
def check_contours_1(img):
	img1 = img.copy()
	img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
	output = check_Color2(img,'Black')
	output1 = check_Color2(img1,'Yellow')
	output2 = check_Color2(img,'White')
	output = sharpen_img(output)
	output1 = sharpen_img(output1)
	output2 = sharpen_img(output2)
	gray = cv2.cvtColor(output1, cv2.COLOR_BGR2GRAY)
	_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
	contours, _ = cv2.findContours(
    	threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	ct_lines=0
	for c in contours:
		area = cv2.contourArea(c)
		# print(area)
		# if area>130000:
		# if area>=3400 and area<=4600:
		if area>60000 and area<70000:
			# print("Entered!")
			# x,y,w,h = cv2.boundingRect(c)
			# if abs(h-35)<=5:
			# 	ct_lines+=1
			# if ct_lines==7:
			# 	return True
			# print(ct_lines)
			return True
	
	return False

def check_contours_2(img,dir):
	img2=img.copy()
	img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
	output_white = check_Color2(img2,'White') #Masking only the white parts of image
	output_black = check_Color2(img,'Black') #Masking only the black part of image
	output_white = sharpen_img(output_white) #Sharpening the blurry image
	output_black = sharpen_img(output_black)



	# Below function can be used for accurately identifying the two black lines in the road.
	for i in range(0,len(output_black)):
		for j in range(0,len(output_black[i])):
			for k in range(0,len(output_black[i][j])):
				if output_black[i][j][k] == 0:
					continue
				else:
					output_black[i][j]=np.array([255,255,255])
					break

	gray1 = cv2.cvtColor(output_white, cv2.COLOR_BGR2GRAY)
	_, threshold1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
	contours1, _ = cv2.findContours(
    	threshold1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# Finding the contours

	flag1=False
	for c in contours1:
		area = cv2.contourArea(c)
		# Detecting the middle white rectangle in road
		if area >=1600 and area<=2700:
			x,y,w,h = cv2.boundingRect(c)
			rect = cv2.minAreaRect(c)
			print("Angle: ",rect[-1])
			if(abs(h-120))<=2:
				if dir=='L':
					# if rect[-1]>=84 and rect[-1]<=90: #If angle of rectangle is between 75 and 90, stop rotating
					# if(abs(rect[-1]-90))%90<5:
					if rect[-1]<=10 or rect[-1]==90:
						flag1=True
				else:
					if rect[-1]<=8 or rect[-1]==90:
						flag1=True
	
		
	return flag1

def check_contours(sim):
	img = return_image(sim)
	img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
	output1 = check_Color2(img2,'White')
	gray = cv2.cvtColor(output1, cv2.COLOR_BGR2GRAY)

# setting threshold of gray image
	_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)



# using a findContours() function
	contours, _ = cv2.findContours(
		threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	ct_lines=1
	flag=False

	for c in contours:
		area = cv2.contourArea(c)
		if area>3400 and area<4500:
			ct_lines+=1
		if ct_lines==7:
			flag=True

	return flag

def return_angle(img,sim,i):
	left_wheel=sim.getObjectHandle('left_joint')
	right_wheel=sim.getObjectHandle('right_joint')
	output_white = check_Color2(img,'White')
	# output2 = check_Color2(img,'Black')
	output_white = sharpen_img(output_white)
	# output_white = output_white[0:512,136:376]
	# output2 = sharpen_img(output2)
	# for i in range(0,len(output2)):
	# 	for j in range(0,len(output2[i])):
	# 		for k in range(0,len(output2[i][j])):
	# 			if output2[i][j][k] == 0:
	# 				continue
	# 			else:
	# 				output2[i][j]=np.array([255,255,255])
	# 				break
	gray1 = cv2.cvtColor(output_white, cv2.COLOR_BGR2GRAY)
	_, threshold1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
	contours1, _ = cv2.findContours(
    	threshold1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# Finding the contours

	flag1=False
	angle=0
	for c in contours1:
		area = cv2.contourArea(c)
		# Detecting the middle white rectangle in road
		if area >=1600 and area<=2700:
			rect = cv2.minAreaRect(c)
			x,y,w,h = cv2.boundingRect(c)
			
			#print("Angle: ",rect[-1])
			
			if(abs(h-120))<=2:
				print("Angle: ",rect[-1],i,sep=" ")

				
				if rect[-1]>=89.5 and rect[-1]<=90: #If angle of rectangle is between 75 and 90, stop rotating
					cv2.rectangle(output_white,(x,y),(x+w,y+h),(0,0,255),1)
					cv2.imwrite(r'C:\Users\Swaroop\OneDrive\Desktop\Swaroop\BMS\EYRC 2022-23 Pharma Bot\PB_Task2_Windows\PB_Task2B_Windows\Img\imga'+str(i)+'.jpg',output_white)
					flag1=True
					sim.setJointTargetVelocity(left_wheel,0)
					sim.setJointTargetVelocity(right_wheel,0)
					angle=rect[-1]
				
	return flag1,angle

def return_angle_2(img,sim,i):
	left_wheel=sim.getObjectHandle('left_joint')
	right_wheel=sim.getObjectHandle('right_joint')
	output_white = check_Color2(img,'White')
	# output2 = check_Color2(img,'Black')
	output_white = sharpen_img(output_white)
	# output_white = output_white[0:512,136:376]
	# output2 = sharpen_img(output2)
	# for i in range(0,len(output2)):
	# 	for j in range(0,len(output2[i])):
	# 		for k in range(0,len(output2[i][j])):
	# 			if output2[i][j][k] == 0:
	# 				continue
	# 			else:
	# 				output2[i][j]=np.array([255,255,255])
	# 				break
	gray1 = cv2.cvtColor(output_white, cv2.COLOR_BGR2GRAY)
	_, threshold1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
	contours1, _ = cv2.findContours(
    	threshold1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# Finding the contours

	flag1=False
	angle=0
	for c in contours1:
		area = cv2.contourArea(c)
		# Detecting the middle white rectangle in road
		if area >=1600 and area<=2700:
			rect = cv2.minAreaRect(c)
			x,y,w,h = cv2.boundingRect(c)
			
			#print("Angle: ",rect[-1])
			
			if(abs(h-120))<=2:
				print("Angle: ",rect[-1],i,sep=" ")

				
				if rect[-1]>=89.5 and rect[-1]<=90: #If angle of rectangle is between 75 and 90, stop rotating
					cv2.rectangle(output_white,(x,y),(x+w,y+h),(0,0,255),1)
					cv2.imwrite(r'C:\Users\Swaroop\OneDrive\Desktop\Swaroop\BMS\EYRC 2022-23 Pharma Bot\PB_Task2_Windows\PB_Task2B_Windows\Img\imga'+str(i)+'.jpg',output_white)
					flag1=True
					sim.setJointTargetVelocity(left_wheel,0)
					sim.setJointTargetVelocity(right_wheel,0)
					angle=rect[-1]
				
	return flag1,angle


			

def check_contours3(img):
	output2 = check_Color2(img,'Black')
	output2 = sharpen_img(output2)

	for i in range(0,len(output2)):
		for j in range(0,len(output2[i])):
			for k in range(0,len(output2[i][j])):
				if output2[i][j][k] == 0:
					continue
				else:
					output2[i][j]=np.array([255,255,255])
					break

	gray = cv2.cvtColor(output2, cv2.COLOR_BGR2GRAY)

# setting threshold of gray image
	_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)



# using a findContours() function
	contours, _ = cv2.findContours(
    	threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	ct_lines=0
	# ct_pixel=0
	flag=False
	# area_list = []
	# pixel_list = []
	# angle=[]

	for c in contours:
		area = cv2.contourArea(c)
		if area>=10000 and area<=11000:
			x,y,w,h = cv2.boundingRect(c)
			img2 = output2[y:y+h,x:x+w]
			ct = check_Color1(img2,'Black')
			ct_lines+=1
			if ct >=400000 and ct<=500000 and ct_lines==2:
				flag=True
	
	return flag


def show_image(img):
    cv2.imshow('',img)
    cv2.waitKey()

def return_image(sim):
	vision_sensor_handle=sim.getObjectHandle('vision_sensor')
	img,res=sim.getVisionSensorImg(vision_sensor_handle) 
	img= np.frombuffer(img, dtype=np.uint8).reshape(res[0],res[1],3)
	img = cv2.flip(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
	return img

def listToString(x):
	s = ''.join(str(ch) for ch in x)
	return s

def selfCorrecting(sim,dir):
	left_wheel=sim.getObjectHandle('left_joint')
	right_wheel=sim.getObjectHandle('right_joint')
	sign=None
	flag=None

	if dir=='L':
		sign=1
	else:
		sign=-1
	
	sim.setJointTargetVelocity(left_wheel,sign*0.035)
	sim.setJointTargetVelocity(right_wheel,-sign*0.035)

	while True:
		img = return_image(sim)
		flag = return_angle(img,sim)
		if flag==True:
			sim.setJointTargetVelocity(left_wheel,0)
			sim.setJointTargetVelocity(right_wheel,0)
			break


def turn(sim,dir):
	left_wheel=sim.getObjectHandle('left_joint')
	right_wheel=sim.getObjectHandle('right_joint')
	sign=None
	flag=None
	# if(dir=='L'):
	# 	sign=-1
	# else:
	# 	sign=1
	
	sim.setJointTargetVelocity(left_wheel,-1)   #0.1
	sim.setJointTargetVelocity(right_wheel,1)  
	#time.sleep(2.5) #for 0.1
	time.sleep(1.1) #1.7 #0.7 
	sim.setJointTargetVelocity(left_wheel,-0.035)   #0.1
	sim.setJointTargetVelocity(right_wheel,0.035) 
	i=0
	prev=-1
	while True:
		img = return_image(sim)
		flag,anglee=return_angle(img,sim,i)
		i=i+1
		if prev==89 or prev==0:
			if flag:
				sim.setJointTargetVelocity(left_wheel,0)
				sim.setJointTargetVelocity(right_wheel,0)
				break
		prev=int(anglee)
def turn_2(sim,dir):
	left_wheel=sim.getObjectHandle('left_joint')
	right_wheel=sim.getObjectHandle('right_joint')
	sign=None
	flag=None

	# if(dir=='L'):
	# 	sign=-1
	# else:
	# 	sign=1

	sim.setJointTargetVelocity(left_wheel,1)   #0.1
	sim.setJointTargetVelocity(right_wheel,-1)  
	#time.sleep(2.5) #for 0.1
	time.sleep(1.1) #1.7 #0.7

	sim.setJointTargetVelocity(left_wheel,0.035)
	sim.setJointTargetVelocity(right_wheel,-0.035)
	i=0
	prev=-1
	
	while True:
		# img = return_image(sim)
		# cv2.imwrite('/home/preetham/eYantra/PB_Task2B_Ubuntu/Sensor_View_A/SV_' + str(i) + ".jpg",img)
		# i=i+1
		img = return_image(sim)
		flag,anglee=return_angle_2(img,sim,i)
		i=i+1
		if prev==0 or prev==89:
			if flag:
				sim.setJointTargetVelocity(left_wheel,0)
				sim.setJointTargetVelocity(right_wheel,0)
				break
		prev=int(anglee)



##############################################################

def control_logic(sim):
	"""
	Purpose:
	---
	This function should implement the control logic for the given problem statement
	You are required to make the robot follow the line to cover all the checkpoints
	and deliver packages at the correct locations.

	Input Arguments:
	---
	`sim`    :   [ object ]
		ZeroMQ RemoteAPI object

	Returns:
	---
	None

	Example call:
	---
	control_logic(sim)
	"""
	##############  ADD YOUR CODE HERE  ##############
	left_wheel=sim.getObjectHandle('left_joint')
	right_wheel=sim.getObjectHandle('right_joint')
	sim.setJointTargetVelocity(left_wheel,1.6)
	sim.setJointTargetVelocity(right_wheel,1.615)
	i=0
	keys = list(checkpoints.keys())
	ch = keys[0]
	# turn(sim)


	while True:
		# turn(sim,'R')

		# img = return_image(sim)
		# cv2.imwrite('/home/preetham/eYantra/PB_Task2B_Ubuntu/Sensor_View_2B/SV_' + str(i) + ".jpg",img)
		# i=i+1
		# turn(sim)
		# if ch=='S':
		# 	sim.setJointTargetVelocity(left_wheel,0)
		# 	sim.setJointTargetVelocity(right_wheel,0)
		# 	break

		img = return_image(sim)
		
		flag1 = check_contours_1(img) 
		if flag1 == True:
			if ch=='S':
				sim.setJointTargetVelocity(left_wheel,0)
				sim.setJointTargetVelocity(right_wheel,0)
				break
			# if ch=='E':
			# 	sim.setJointTargetVelocity(left_wheel,0)
			# 	sim.setJointTargetVelocity(right_wheel,0)
			# 	cv2.imwrite('/home/preetham/eYantra/PB_Task2B_Ubuntu/Sensor_View_2B/SV_' + str(i) + ".jpg",img)
			# 	selfCorrecting(sim,checkpoints[ch][1])

			# else:
			time.sleep(0.646) #0.7 #0.35
			sim.setJointTargetVelocity(left_wheel,0)
			sim.setJointTargetVelocity(right_wheel,0)
			print("Checkpoint ",ch," Reached")

			if checkpoints[ch][1]!='Y':
				# if ch=='A':
				# 	turn_2(sim,checkpoints[ch][1])
				# else:
				if checkpoints[ch][1]=='L':
					turn(sim,'L')
				else:
					turn_2(sim,'R')
				#selfCorrecting(sim,checkpoints[ch][1])

			else:
				# print("Qr code region detected")
				# print("Qr code region detected")
				sim.setJointTargetVelocity(left_wheel,0)
				sim.setJointTargetVelocity(right_wheel,0)
				qr_message=None
				while True:
			# time.sleep(1)
				# img = return_image(sim)
			# cv2.imwrite('/home/preetham/eYantra/PB_Task2B_Ubuntu/Test_Images/QR_before/qrb_' + ch + ".jpg",img)
					arena_dummy_handle = sim.getObject("/Arena_dummy")
					childscript_handle = sim.getScript(sim.scripttype_childscript, arena_dummy_handle, "")
					st1 = "checkpoint " + ch
					print(sim.callScriptFunction("activate_qr_code", childscript_handle, st1),"huhuhu",sep=" ")
					sim.setJointTargetVelocity(left_wheel,0)
					sim.setJointTargetVelocity(right_wheel,0)
					# time.sleep(1)
					#qr_message=read_qr_code(sim)

					print("QR message at ",ch,": ",qr_message)
					img = return_image(sim)
					if qr_message!=None:
					# cv2.imwrite('/home/preetham/eYantra/PB_Task2B_Ubuntu/Test_Images/QR_codes/qr_' + ch + ".jpg",img)
						arena_dummy_handle = sim.getObject("/Arena_dummy")
						childscript_handle = sim.getScript(sim.scripttype_childscript, arena_dummy_handle, "")
						sim.callScriptFunction("deliver_package", childscript_handle, shapes_qr[qr_message], "checkpoint E")
						arena_dummy_handle = sim.getObject("/Arena_dummy")
						childscript_handle = sim.getScript(sim.scripttype_childscript, arena_dummy_handle, "")
						st2 = "checkpoint " + ch
						sim.callScriptFunction("deactivate_qr_code", childscript_handle, st2)
						break
					
			
			sim.setJointTargetVelocity(left_wheel,0)
			sim.setJointTargetVelocity(right_wheel,0)
			if checkpoints[ch][1] == 'L':
				sim.setJointTargetVelocity(left_wheel,1.6)
				sim.setJointTargetVelocity(right_wheel,1.615)
			elif checkpoints[ch][1] == 'R':
				sim.setJointTargetVelocity(left_wheel,1.6)
				sim.setJointTargetVelocity(right_wheel,1.615)
			else:
				sim.setJointTargetVelocity(left_wheel,1.6)
				sim.setJointTargetVelocity(right_wheel,1.6)
			keys.pop(0)
			ch=keys[0]



		

					







	##################################################

def read_qr_code(sim):
	"""
	Purpose:
	---
	This function detects the QR code present in the camera's field of view and
	returns the message encoded into it.

	Input Arguments:
	---
	`sim`    :   [ object ]
		ZeroMQ RemoteAPI object

	Returns:
	---
	`qr_message`   :    [ string ]
		QR message retrieved from reading QR code

	Example call:
	---
	control_logic(sim)
	"""
	qr_message = None
	##############  ADD YOUR CODE HERE  ##############
	left_wheel=sim.getObjectHandle('left_joint')
	right_wheel=sim.getObjectHandle('right_joint')
	sim.setJointTargetVelocity(left_wheel,0)
	sim.setJointTargetVelocity(right_wheel,0)
	sim.setJointTargetVelocity(left_wheel,0.025)
	sim.setJointTargetVelocity(right_wheel,0.025)
	flag=False
	while True:
		image = return_image(sim)
		for qrcode in decode(image):
			qr_message = qrcode.data.decode('utf-8')
			print("Qr message: ",qr_message)
			if qr_message!=None:
				flag=True
				break
		
		if flag:
			time.sleep(3.5) #10
			sim.setJointTargetVelocity(left_wheel,0)
			sim.setJointTargetVelocity(right_wheel,0)
			break
	##################################################
	return qr_message


######### YOU ARE NOT ALLOWED TO MAKE CHANGES TO THE MAIN CODE BELOW #########

if __name__ == "__main__":
	client = RemoteAPIClient()
	sim = client.getObject('sim')	

	try:

		## Start the simulation using ZeroMQ RemoteAPI
		try:
			return_code = sim.startSimulation()
			if sim.getSimulationState() != sim.simulation_stopped:
				print('\nSimulation started correctly in CoppeliaSim.')
			else:
				print('\nSimulation could not be started correctly in CoppeliaSim.')
				sys.exit()

		except Exception:
			print('\n[ERROR] Simulation could not be started !!')
			traceback.print_exc(file=sys.stdout)
			sys.exit()

		## Runs the robot navigation logic written by participants
		try:
			time.sleep(5)
			control_logic(sim)

		except Exception:
			print('\n[ERROR] Your control_logic function throwed an Exception, kindly debug your code!')
			print('Stop the CoppeliaSim simulation manually if required.\n')
			traceback.print_exc(file=sys.stdout)
			print()
			sys.exit()

		
		## Stop the simulation using ZeroMQ RemoteAPI
		try:
			return_code = sim.stopSimulation()
			time.sleep(0.5)
			if sim.getSimulationState() == sim.simulation_stopped:
				print('\nSimulation stopped correctly in CoppeliaSim.')
			else:
				print('\nSimulation could not be stopped correctly in CoppeliaSim.')
				sys.exit()

		except Exception:
			print('\n[ERROR] Simulation could not be stopped !!')
			traceback.print_exc(file=sys.stdout)
			sys.exit()

	except KeyboardInterrupt:
		## Stop the simulation using ZeroMQ RemoteAPI
		return_code = sim.stopSimulation()
		time.sleep(0.5)
		if sim.getSimulationState() == sim.simulation_stopped:
			print('\nSimulation interrupted by user in CoppeliaSim.')
		else:
			print('\nSimulation could not be interrupted. Stop the simulation manually .')
			sys.exit()
