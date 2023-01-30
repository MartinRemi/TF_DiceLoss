import tensorflow as tf

def project(inputs):
	X = inputs[:, 0]
	Y = inputs[:, 1]
	Z = inputs[:, 2]

	h = 224.0 * tf.divide(X, Z) + 112.0
	w = 224.0 * tf.divide(Y, Z) + 112.0
	h = tf.minimum(tf.maximum(h, 0), 223)
	w = tf.minimum(tf.maximum(w, 0), 223)
	indices = tf.stack([h,w], 1)

	return indices

def DICE_img(gt, pred, smooth = 1):
	gt_flat = tf.reshape(gt, [-1])
	pred_flat = tf.reshape(pred, [-1])
	intersection= tf.reduce_mean(gt_flat * pred_flat)
	return (2.0 * intersection + smooth) / (tf.reduce_mean(gt_flat) + K.reduce_mean(pred_flat) + smooth)

def DICE_pts(gt, pred, smooth = 1):
	# intersection = tf.sets.set_union(tf.cast(gt, tf.int64), tf.cast(pred, tf.int64))
	# print(intersection.eval())
	gt64, _ = tf.unique(tf.bitcast(gt, type=tf.int64))
	pred64, _ = tf.unique(tf.bitcast(pred, type=tf.int64))
	all64 = tf.concat([gt64, pred64], 0)
	uniques64, _, unique_count = tf.unique_with_counts(all64)
	pairs = tf.greater_equal(unique_count, tf.constant([2]))
	return (tf.constant(2, dtype=tf.int32) * tf.reduce_sum(tf.cast(pairs, tf.int32))) / (tf.shape(gt64)[0] + tf.shape(pred64)[0])

def DICE_loss(gt, pred):
	print(gt.shape)
	print(pred.shape)
	gt_proj = project(gt)
	pred_proj = project(pred)

	dice_score = DICE(gt_proj, pred_proj)
	return 1-dice_score

def PROJ_loss(gt, pred):
	gt_proj = project(gt)
	pred_proj = project(pred)
	pred_pred = tf.tensordot(pred_proj, tf.transpose(pred_proj), axes=1)
	I = tf.eye(tf.shape(pred_pred)[0], tf.shape(pred_pred)[0])
	pred_gt = tf.tensordot(pred_proj, tf.transpose(gt_proj), axes=1)
	return tf.reduce_sum(tf.reduce_min(tf.abs(tf.subtract(tf.reduce_sum(pred_pred*I, axis=0), pred_gt)), axis=1))

def DICE_proj_loss(gt, pred):
	gt_proj = project(gt)
	pred_proj = project(pred)

	minx = tf.reduce_min(gt[:, 0])
	maxx = tf.reduce_max(gt[:, 0])
	miny = tf.reduce_min(gt[:, 1])
	maxy = tf.reduce_max(gt[:, 1])
	minz = tf.reduce_min(gt[:, 2])
	maxz = tf.reduce_max(gt[:, 2])

	center = [(minx + (maxx-minx) / 2), (miny + (maxy-miny) / 2), (minz + (maxz-minz) / 2)]
	center = tf.convert_to_tensor(center, dtype=tf.float32)

	rotation = [90, 0, 0]
	Rx = np.array([[1, 0, 0], [0, np.cos(rotation[0]), -np.sin(rotation[0])], [0, np.sin(rotation[0]), np.cos(rotation[0])]])
	Ry = np.array([[np.cos(rotation[1]), 0, np.sin(rotation[1])], [0, 1, 0], [-np.sin(rotation[1]), 0, np.cos(rotation[1])]])
	Rz = np.array([[np.cos(rotation[2]), -np.sin(rotation[2]), 0], [np.sin(rotation[2]), np.cos(rotation[2]), 0], [0, 0, 1]])

	rot = np.dot(np.dot(Rx, Ry), Rz)
	rot = tf.convert_to_tensor(rot, dtype=tf.float32)

	gt_90 = gt - center
	pred_90 = pred - center

	gt_90 = tf.matmul(gt_90, rot)
	pred_90 = tf.matmul(pred_90, rot)

	return interFunction(gt_proj, pred_proj) + interFunction(gt_90, pred_90)

def interFunction(gt_proj, pred_proj):
	AA = tf.matmul(gt_proj, tf.transpose(gt_proj))
	AB = tf.matmul(gt_proj, tf.transpose(pred_proj))
	I = tf.eye(tf.shape(AA)[0], tf.shape(AA)[0])
	diff = tf.abs(tf.subtract(tf.reduce_sum(AA*I, axis=1, keep_dims=True), AB))
	penalty = tf.reduce_min(diff, axis=1)
	penalty = penalty / (penalty + 0.000000001) # avoids division by zero
	penalty = 1 / (1 + tf.exp(-30 * (penalty - 0.5)))
	penalty = tf.reduce_sum(1 - penalty)

	return penalty / tf.reduce_sum(I)

if __name__=='__main__':
	import numpy as np
	with tf.Session('') as sess:
		sess.run(tf.initialize_all_variables())
		gt_pts = np.array([[255.0, 255.0, 10.0], [123.0, 300.2, 121.1], [17.3, 182.1, 167.1], [1.0, 2.0, 6.0]])
		pred_pts = np.array([[255.0, 255.0, 10.0], [123.0, 214.2, 124.1], [17.3, 180.1, 168.1], [3.0, 4.0, 6.0]])

		gt_pts_tf = tf.convert_to_tensor(gt_pts, dtype=tf.float32)
		pred_pts_tf = tf.convert_to_tensor(pred_pts, dtype=tf.float32)
		loss = DICE_proj_loss(gt_pts_tf, pred_pts_tf)
		diceloss_run =sess.run([loss])
		print(diceloss_run)
