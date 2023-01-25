from scipy import spatial


class Cluster(object):

    def __init__(self):
        self.data = []


class DBScan(object):

    def __init__(self, epsilon, min_points, similarity_index):
        """
        :param epsilon: for the epsilon neighborhood
        :param min_points: the minimum number of points within the epsilon neighborhood
        :param similarity_index: for checking of gene expressions similarity
        :return: None
        """
        self.epsilon = epsilon
        self.min_points = min_points
        self.similarity_index = similarity_index
        self.data = None
        self.clusters = []
        self.processed = set()
        self.outliers = []

    def fit(self, data):
        self.data = data

        for i, point in enumerate(self.data):
            print(i)
            if str(point) in self.processed:
                continue
            self.processed.add(str(point))

            neighbors = self.region_query(point)
            if len(neighbors) >= self.min_points:
                print('CLUSTER!')
                c = Cluster()
                self.clusters.append(c)
                self.expand_cluster(point, neighbors, c)

        self.find_outliers()

    def expand_cluster(self, point, neighbors, cluster):
        cluster.data.append(point)
        for point_ in neighbors:
            if str(point_) not in self.processed:
                self.processed.add(str(point_))
                neighbors_ = self.region_query(point_)
                if len(neighbors_) >= self.min_points:
                    neighbors += neighbors_
                all_cluster_data = []
                for c in self.clusters:
                    all_cluster_data += c.data
                if point_ not in all_cluster_data:
                    cluster.data.append(point_)

    def region_query(self, point):
        neighbor_list = []
        for d in self.data:
            if self.euclidean_distance(point[0], d[0]) <= self.epsilon and 1 - spatial.distance.cosine(point[1], d[1]) >= self.similarity_index:
                neighbor_list.append(d)
        return neighbor_list

    def euclidean_distance(self, x, y):
        return sum((p-q)**2 for p, q in zip(x, y)) ** .5

    def find_outliers(self):
        all_cluster_data = []
        for c in self.clusters:
            all_cluster_data += c.data

        self.outliers = [d for d in self.data if d not in all_cluster_data]
