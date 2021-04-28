import numpy as np
import time

class Leaf(object):
    def __init__(self, x, y, z, idx):
        if idx.size == 0:
            self.empty = True
        else:
            self.x = x[0,0,0]
            self.y = y[0,0,0]
            self.z = z[0,0,0]
            self.idx = idx[0,0,0]
            self.empty = False

    def distance_to_center(self, x, y, z):
        if not self.empty:
            return np.sqrt((x - self.x)**2 + (y - self.y)**2 + (z - self.z)**2)
        else:
            return np.inf

    def __str__(self):
        s = ""
        if not self.empty:
            s += "Leaf: \n"
            s += f"    - x:   {self.x}\n"
            s += f"    - y:   {self.y}\n"
            s += f"    - z:   {self.z}\n"
            s += f"    - idx: {self.idx}\n"
        return s


class Node(object):
    '''
     _____________________________
    |              |              |
    |     Q2F      |      Q1f     |
    |              |              |
    |--------------|--------------|
    |              |              |
    |     Q3F      |      Q4F     |
    |______________|______________|
    '''
    def __init__(self, x, y, z, idx):
        THRESHOLD = 1

        Nx, Ny, Nz = x.shape

        cNx, cNy, cNz = Nx // 2, Ny // 2, Nz // 2

        self.xc = x.mean()
        self.yc = y.mean()
        self.zc = z.mean()

        self.x = x
        self.y = y
        self.z = z

        xc = self.xc
        yc = self.yc
        zc = self.zc

        mask_q1f = slice(cNx, Nx), slice(cNy, Ny), slice(cNz, Nz)
        mask_q2f = slice(0,  cNx), slice(cNy, Ny), slice(cNz, Nz)
        mask_q3f = slice(0,  cNx), slice(0,  cNy), slice(cNz, Nz)
        mask_q4f = slice(cNx, Nx), slice(0,  cNy), slice(cNz, Nz)

        mask_q1b = slice(cNx, Nx), slice(cNy, Ny), slice(0, cNz)
        mask_q2b = slice(0,  cNx), slice(cNy, Ny), slice(0, cNz)
        mask_q3b = slice(0,  cNx), slice(0,  cNy), slice(0, cNz)
        mask_q4b = slice(cNx, Nx), slice(0,  cNy), slice(0, cNz)

        if x[mask_q1f].size <= THRESHOLD:
            self.q1f = Leaf(x[mask_q1f], y[mask_q1f], z[mask_q1f], idx[mask_q1f])
        else:
            self.q1f = Node(x[mask_q1f], y[mask_q1f], z[mask_q1f], idx[mask_q1f])

        if x[mask_q2f].size <= THRESHOLD:
            self.q2f = Leaf(x[mask_q2f], y[mask_q2f], z[mask_q2f], idx[mask_q2f])
        else:
            self.q2f = Node(x[mask_q2f], y[mask_q2f], z[mask_q2f], idx[mask_q2f])

        if x[mask_q3f].size <= THRESHOLD:
            self.q3f = Leaf(x[mask_q3f], y[mask_q3f], z[mask_q3f], idx[mask_q3f])
        else:
            self.q3f = Node(x[mask_q3f], y[mask_q3f], z[mask_q3f], idx[mask_q3f])

        if x[mask_q4f].size <= THRESHOLD:
            self.q4f = Leaf(x[mask_q4f], y[mask_q4f], z[mask_q4f], idx[mask_q4f])
        else:
            self.q4f = Node(x[mask_q4f], y[mask_q4f], z[mask_q4f], idx[mask_q4f])

        if x[mask_q1b].size <= THRESHOLD:
            self.q1b = Leaf(x[mask_q1b], y[mask_q1b], z[mask_q1b], idx[mask_q1b])
        else:
            self.q1b = Node(x[mask_q1b], y[mask_q1b], z[mask_q1b], idx[mask_q1b])

        if x[mask_q2b].size <= THRESHOLD:
            self.q2b = Leaf(x[mask_q2b], y[mask_q2b], z[mask_q2b], idx[mask_q2b])
        else:
            self.q2b = Node(x[mask_q2b], y[mask_q2b], z[mask_q2b], idx[mask_q2b])

        if x[mask_q3b].size <= THRESHOLD:
            self.q3b = Leaf(x[mask_q3b], y[mask_q3b], z[mask_q3b], idx[mask_q3b])
        else:
            self.q3b = Node(x[mask_q3b], y[mask_q3b], z[mask_q3b], idx[mask_q3b])

        if x[mask_q4b].size <= THRESHOLD:
            self.q4b = Leaf(x[mask_q4b], y[mask_q4b], z[mask_q4b], idx[mask_q4b])
        else:
            self.q4b = Node(x[mask_q4b], y[mask_q4b], z[mask_q4b], idx[mask_q4b])


    def _get_nearest_neighbor(self, x, y, z):
        return self._get_nearest_neighbor_best_first(x,y,z)


    def _get_nearest_neighbor_depth_first(self, x, y, z):
        done = False
        current_best = self
        best_distance = 2147483647.0
        children = self.get_childs()
        dx = np.mean(self.x[1:self.x.shape[0],:,0] - self.x[:self.x.shape[0]-1,:,0])
        dy = np.mean(self.y[:,1:self.x.shape[1],0] - self.y[:,:self.x.shape[1]-1,0])
        dz = np.mean(self.z[0,0,1:self.x.shape[2]] - self.z[0,0,self.x.shape[2]-1])
        optimal = np.sqrt(dx**2 + dy**2 + dz**2)
        for child in children:
            if isinstance(child, Leaf):
                distance = child.distance_to_center(x,y,z)
                if distance < best_distance:
                    best_distance = distance
                    current_best = child
                if distance <= (optimal + 1.19e-07):
                    return child
            else:
                c = child._get_nearest_neighbor_depth_first(x, y, z)
                distance = c.distance_to_center(x,y,z)
                if distance < best_distance:
                    best_distance = distance
                    current_best = c
                if distance <= (optimal + 1.19e-07):
                    return c
        return current_best


    def _get_nearest_neighbor_best_first(self, x, y, z):
        done = False
        current_best = self
        while not done:
            options = current_best.get_childs()
            scores = current_best.get_scores(x, y, z)
            current_best = options[np.argmin(scores)]
            if isinstance(current_best, Leaf):
                return current_best
        return

    def get_scores(self, x, y, z):
        return np.array([q.distance_to_center(x, y, z) for q in self.get_childs()])

    def distance_to_center(self, x, y, z):
        return np.sqrt((x - self.xc)**2 + (y - self.yc)**2 + (z - self.zc)**2)

    def get_childs(self):
        return (self.q1f, self.q2f, self.q3f, self.q4f, self.q1b, self.q2b, self.q3b, self.q4b)


class Octree(object):
    '''
     _____________________________
    |              |              |
    |     Q2F      |      Q1f     |
    |              |              |
    |--------------|--------------|
    |              |              |
    |     Q3F      |      Q4F     |
    |______________|______________|
    '''

    def __init__(self, xvol, yvol, zvol):
        assert (xvol.size == yvol.size) and (yvol.size == zvol.size)
        self._idx = np.arange(xvol.size, dtype=int).reshape(xvol.shape)
        self._tree = Node(xvol, yvol, zvol, self._idx)

    def get_nearest_neighbor(self, x, y, z):
        return self._tree._get_nearest_neighbor(x, y, z)

    def get_nearest_neighbor_depth_first(self, x, y, z):
        return self._tree._get_nearest_neighbor_depth_first(x, y, z)

    def get_nearest_neighbor_best_first(self, x, y, z):
        return self._tree._get_nearest_neighbor_best_first(x, y, z)


def naive_search(x,y,z,id,xt,yt,zt):
    Nx, Ny, Nz = x.shape
    best = (np.nan, np.nan, np.nan, np.nan)
    for ii in range(1, Nx-1):
        for jj in range(1, Ny-1):
            for kk in range(1, Nz-1):
                xbool = x[ii-1,jj,kk] <= xt <= x[ii+1,jj,kk]
                ybool = y[ii,jj-1,kk] <= yt <= y[ii,jj+1,kk]
                zbool = z[ii,jj,kk-1] <= zt <= z[ii,jj,kk+1]
                if xbool and ybool and zbool:
                    score = np.inf
                    for ix in range(ii-1, ii+2):
                        for jy in range(jj-1, jj+2):
                            for kz in range(kk-1, kk+2):
                                distance = np.sqrt((xt - x[ix,jy,kz])**2 + (yt - y[ix,jy,kz])**2 + (zt - z[ix,jy,kz])**2)
                                if distance < score:
                                    score = distance
                                    best = x[ix,jy,kz], y[ix,jy,kz], z[ix,jy,kz], id[ix,jy,kz]
                    return best

if __name__ == "__main__":

    ZPG = True
    CONCAVE = not ZPG

    if CONCAVE:
        import h5py
        coord_path = './mean_flow.nn.h5'
        t1 = time.time()
        with h5py.File(coord_path, 'r') as hf:
            coordinates = hf["coordinates"][:]
        t2 = time.time()
        print(f"Time Elapsed Reading Coordinates: {t2 - t1} s")
        x,y,z = coordinates[:,:,0,:], coordinates[:,:,1,:], coordinates[:,:,2,:]
        t3 = time.time()
        print(f"Time Elapsed Building Octree {t3 - t2} s")

        print(tree.get_nearest_neighbor_best_first(-0.2, 0.005, 0.05))
        print(tree.get_nearest_neighbor_best_first(-0.1, 0.005, 0.05))
        print(tree.get_nearest_neighbor_best_first(-0.05, 0.05, 0.07))
        print(tree.get_nearest_neighbor_best_first(0.48, 0.11, 0.06))
        print(tree.get_nearest_neighbor_best_first(1.35, 0.25, 0.08))
        t4 = time.time()
        print(f"Time Elapsed in best-first NNS: {t4 - t3} s")
        t3 = time.time()
        print(tree.get_nearest_neighbor_depth_first(-0.2, 0.005, 0.05))
        print(tree.get_nearest_neighbor_depth_first(-0.1, 0.005, 0.05))
        print(tree.get_nearest_neighbor_depth_first(-0.05, 0.05, 0.07))
        print(tree.get_nearest_neighbor_depth_first(0.48, 0.11, 0.06))
        print(tree.get_nearest_neighbor_depth_first(1.35, 0.25, 0.08))
        t4 = time.time()
        print(f"Time Elapsed in depth-first NNS: {t4 - t3} s")
        t3 = time.time()
        print(naive_search(x, y, z, tree._idx, -0.2, 0.005, 0.05))
        print(naive_search(x, y, z, tree._idx, -0.1, 0.005, 0.05))
        print(naive_search(x, y, z, tree._idx, -0.05, 0.05, 0.07))
        print(naive_search(x, y, z, tree._idx, 0.48, 0.11, 0.06))
        print(naive_search(x, y, z, tree._idx, 1.35, 0.25, 0.08))
        t4 = time.time()
        print(f"Time Elapsed in Naive Search: {t4 - t3} s")
        print(f"Total Time Elapsed: {t4 - t1} s")
    elif ZPG:
        import pandas as pd
        coord_path = "./coord_1_440.txt"
        coordinates = pd.read_csv(coord_path, engine='c', header=None, sep='\s+').values.reshape((440, 60, 80, 3))
        x,y,z = coordinates[:,:,:,0], coordinates[:,:,:,1], coordinates[:,:,:,2]

    Nx, Ny, Nz = x.shape

    tree = Octree(x, y, z)


    time_best = []
    time_naive = []

    for ii in range(1, Nx-1):
        jj = np.random.randint(1, Ny-1)
        kk = np.random.randint(1, Nz-1)
        xt = np.around(x[ii, jj, kk], decimals=3)
        yt = np.around(y[ii, jj, kk], decimals=3)
        zt = np.around(z[ii, jj, kk], decimals=3)
        t1 = time.time()
        tree.get_nearest_neighbor_best_first(xt, yt, zt)
        t2 = time.time()
        time_best.append(t2 - t1)
        t1 = time.time()
        naive_search(x, y, z, tree._idx, xt, yt, zt)
        t2 = time.time()
        time_naive.append(t2 - t1)

    time_best = np.array(time_best)
    time_naive = np.array(time_naive)
    total_time_best = np.sum(time_best)
    total_time_naive = np.sum(time_naive)
    mean_time_best = np.mean(time_best)
    mean_time_naive = np.mean(time_naive)
    median_time_best = np.median(time_best)
    median_time_naive = np.median(time_naive)
    sigma_time_best = np.std(time_best)
    sigma_time_naive = np.std(time_naive)

    print(f"Total Time Best-First:       {total_time_best} s")
    print(f"Mean Time Best-First:        {mean_time_best} s")
    print(f"Median Time Best-First:      {median_time_best} s")
    print(f"STDDEV Best-First:           {sigma_time_best} s")

    print(f"Total Time Naive:       {total_time_naive} s")
    print(f"Mean Time Naive:        {mean_time_naive} s")
    print(f"Median Time Naive:      {median_time_naive} s")
    print(f"STDDEV Naive:           {sigma_time_naive} s")
