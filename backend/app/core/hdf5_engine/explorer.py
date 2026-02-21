import h5py


def build_tree(hdf, path="/"):
    nodes = []

    for key in hdf[path]:
        full_path = path + key

        if isinstance(hdf[full_path], h5py.Group):
            nodes.append({
                "name": key,
                "path": full_path,
                "type": "group",
                "children": build_tree(hdf, full_path + "/")
            })
        else:
            dataset = hdf[full_path]
            nodes.append({
                "name": key,
                "path": full_path,
                "type": "dataset",
                "shape": dataset.shape,
                "dtype": str(dataset.dtype)
            })

    return nodes
