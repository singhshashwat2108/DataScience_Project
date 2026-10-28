import sys, os; sys.path.insert(0, os.getcwd()); import warnings, numpy as np, torch; warnings.filterwarnings("ignore")
torch.set_num_threads(int(sys.argv[1]))
import gnn_segmentation.gnn as G
from gnn_segmentation.data import prepare_cc_full
from gnn_segmentation.graph import knn_graph, to_edge_index
from gnn_segmentation.evaluate import cluster
s=prepare_cc_full(); ei=to_edge_index(knn_graph(s.X,5)); tr=s.idx('train')
for kind in ('sage','gcn'):
    out=[G.train_gnn(s.X,ei,tr,kind,'vgae',32,seed=3,epochs=60)[0] for _ in range(2)]
    L=[cluster(z,tr,4,'kmeans',3)[0] for z in out]
    print(sys.argv[1],'threads',kind,'bit_identical',np.array_equal(out[0],out[1]),'maxdiff',float(np.abs(out[0]-out[1]).max()),'labels_identical',np.array_equal(L[0],L[1]))
a=[G.train_autoencoder(s.X,tr,s.idx('val'),seed=3,epochs=20) for _ in range(2)]
print('AE bit_identical',np.array_equal(a[0],a[1]))
