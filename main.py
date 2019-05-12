import pygame
import numpy as np
import math

def intersect(line, plane):
	w = line[1] - plane[0]
	s = -plane[1].dot(w)/plane[1].dot(line[0])
	return w + s*line[0] + plane[0] 

def clip_to_plane(poly, plane):
	if len(poly) != 3:
		return poly
	clipped_verts = []
	for v in range(0, 3):
		d = poly[v] - plane[0]
		if d.dot(plane[1]) < 0:
			print("Clipping occured")
			clipped_verts.append(v)
		
	inside = list(set([0, 1, 2]) - set(clipped_verts))	
	if len(clipped_verts) == 3:
		# Triangle completely outside
		return [] 
	elif len(clipped_verts) == 2:
		p = []
		inside = inside[0]
		# Add the new points in order
		for i in range(0, 3):
			if i in clipped_verts:
				l = poly[clipped_verts[i]] - poly[inside]
				l = l/np.linalg.norm(l)
				p.append(intersect([l, poly[clipped_verts[i]]], plane))
			else:
				p.append(poly[i])
		return p
	elif len(clipped_verts) == 1:
		p = []
		print(inside)
		for i in range(0, 3):
			if i in inside:
				l = poly[clipped_verts[0]] - poly[i]
				l = l/np.linalg.norm(l)
				p.append(poly[i])
				p.append(intersect([l, poly[clipped_verts[0]]], plane))
			else:
				pass
				#p.append(poly[i])
		# Now split the rect to two tri
		return [*np.array([p[0], p[1], p[3]]), *np.array([p[0], p[2], p[3]])]
	return poly

class Marine:
	def __init__(self):
		self._pos = np.array([0, 0, 0, 0])

	@property
	def pos(self):
		return self._pos

class Wall:
	def __init__(self, x=0, y=0, z=30):
		self._pos = np.array([x, y, z])
		self.vertices = np.array((
			[-1, -1, 0],
			[-1, 1, 0],
			[1, 1, 0]
		))

		self.edges = [0, 1, 2]

		self._theta = 0
		self._scale = 10

	@property
	def pos(self):
		return self._pos

	@pos.setter
	def pos(self, pos):
		self._pos = pos
	
	def transform(self):
		tc = []

		transl = np.array(
			([1, 0, 0, self._pos[0]],
			[0, 1, 0, self._pos[1]],
			[0, 0, 1, self._pos[2]],
			[0, 0, 0, 1]))

		scale = np.array(
			([self._scale, 0, 0, 1],
			[0, self._scale, 0, 1],
			[0, 0, self._scale, 1],
			[0, 0, 0, 1]))

		rot = np.array((
			[math.cos(self._theta),  0, math.sin(self._theta), 1],
			[0,                1,               0, 1],
			[-math.sin(self._theta), 0, math.cos(self._theta), 1],
			[0, 0, 0, 1]
			))

		for coord in self.vertices:
			coord = np.array([*coord, 1]) # Transform to honogenous coord
			s = scale.dot(coord)
			r = rot.dot(s)
			p = transl.dot(r)			
			tc.append(p)
		return tc

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

if __name__ == '__main__':
	(width, height) = (600, 400)
	screen = pygame.display.set_mode((width, height))
	pygame.display.set_caption('PyDoom')
	w = Wall(0, 0, 30)

	running = True
	yaw = 0
	pitch = 0

	eye = np.array([0, 0, 10], dtype=np.float64)
	up = np.array(([0, 1, 0]))
	target = np.array(([0, 0, 20]))


	fov = 120
	
	far = 10
	near = 1
	leftdown = False
	rightdown = False
	updown = False
	downdown = False

	while running:
		screen.fill((0,0,0))
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_LEFT:
					leftdown = True
				if event.key == pygame.K_RIGHT:
					rightdown = True
				if event.key == pygame.K_UP:
					updown = True
				if event.key == pygame.K_DOWN:
					downdown = True
			elif event.type == pygame.KEYUP:
				if event.key == pygame.K_LEFT:
					leftdown = False
				if event.key == pygame.K_RIGHT:
					rightdown = False
				if event.key == pygame.K_UP:
					updown = False
				if event.key == pygame.K_DOWN:
					downdown = False
		if leftdown:
			yaw += 0.1
		if rightdown:
			yaw -= 0.1

		eyeR = np.array((
			[math.cos(yaw), 0, math.sin(yaw)],
			[0, 1, 0],
			[-math.sin(yaw), 0, math.cos(yaw)]
			))
		if updown:
			eye += eyeR.dot(np.array(([0, 0, -0.5])))
			#eye[2] += 1
		if downdown:
			eye += eyeR.dot(np.array(([0, 0, 0.5])))
			#eye[2] -= 1

		S = 1/math.tan((fov/2)*(math.pi/180))

		projection = np.array((
		[S*2/3, 0, 0, 0],
		[0, S, 0, 0],
		[0, 0, - (far+near)/(far-near), - 2*far*near/(far-near)],
		[0, 0, -1, 0]
		))

		cosPitch = math.cos(pitch)
		sinPitch = math.sin(pitch)
		cosYaw = math.cos(yaw)
		sinYaw = math.sin(yaw)

		xaxis = np.array(([cosYaw, 0, -sinYaw]))
		yaxis = np.array(([sinYaw*sinPitch, cosPitch, cosYaw*sinPitch]))
		zaxis = np.array(([sinYaw*cosPitch, -sinPitch, cosPitch*cosYaw]))
		V = np.array((
			[*xaxis, -xaxis.dot(eye)],
			[*yaxis, -yaxis.dot(eye)],
			[*zaxis, -zaxis.dot(eye)],
			[0, 0, 0, 1]
			))
		r = w.transform()
		out = []

		
		# Go thtough vertices one tri at a time
		offset = 0
		while offset < len(w.edges):
			tri = []
			cliplist = []
			ndc = []
			for i in range(0, 3):
				edge = w.edges[i+offset]
				# Get ndc coords for this tri
				coord = V.dot(np.array([*w.vertices[edge], 1]))
				wc = projection.dot(coord)
				ndc.append(np.array([wc[0]/wc[3], wc[1]/wc[3], wc[2]/wc[3], 1]))


			# Right
			ndc = clip_to_plane(ndc, (np.array([1, 0, 0, 1]), np.array([-1, 0, 0, 0])))
			# Left
			ndc = clip_to_plane(ndc, (np.array([-1, 0, 0, 1]), np.array([1, 0, 0, 0])))
			# Top
			ndc = clip_to_plane(ndc, (np.array([0, 0.5, 0, 1]), np.array([0, -1, 0, 0])))
			# Bottom
			ndc = clip_to_plane(ndc, (np.array([0, -0.5, 0, 1]), np.array([0, 1, 0, 0])))			
			# Far
			ndc = clip_to_plane(ndc, (np.array([0, 0, 1, 1]), np.array([0, 0, -1, 0])))
			# Near
			ndc = clip_to_plane(ndc, (np.array([0, 0, -1, 1]), np.array([0, 0, 1, 0])))

			for i in range(0, len(ndc)):
				screen_coord = ((ndc[i][0] + 1) * width / 2 - w.vertices[edge][0], (ndc[i][1] + 1) * height / 2 - w.vertices[edge][1])
				tri.append(screen_coord)
	

			#else:
			out.extend(list(map(lambda c: (c[0], c[1]), tri)))
			offset += 3
			cliplist = []

		# Working
		#for edge in w.edges:
		#	coord = V.dot(np.array([*w.vertices[edge], 1]))
		#	wc = projection.dot(coord)
		#	ndc = np.array([wc[0]/wc[3], wc[1]/wc[3], wc[2]/wc[3], 1])
		#	if not -1 <= ndc[0] <= 1:
		#		print("Clipping in x")
		#	screen_coord = ((ndc[0] + 1) * width / 2 - w.vertices[edge][0], (ndc[1] + 1) * height / 2 - w.vertices[edge][1])
		#	out.append((screen_coord[0], screen_coord[1]))
		if len(out) > 0:
			pygame.draw.polygon(screen, (0, 255, 128), out, 2)
		pygame.display.flip()