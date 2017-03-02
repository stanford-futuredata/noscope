
import noscope
from noscope.CommonUtils import mkdir_p
from noscope.filters.BlockedHistogram import BlockedHistogram
from noscope.filters.BlockedRawImage import BlockedRawImage
import argparse
import os
import time
import numpy as np

def find_ref_index(Y_train, num_consec=60):
    count = 0
    for i, label in enumerate(Y_train):
        if label == 0:
            count += 1
        else:
            # reset
            count = 0
        if count >= num_consec:
            return i
    return -1

def get_distances(filter, DELAY, ref_index, metric, X, use_max):
    if ref_index == -1:
        return np.array([filter.compute_blocked_distances(i, metric, block1,
            block2, use_max=use_max) for i, (block1, block2) in enumerate(zip(X[DELAY:],
                X[:-DELAY]))])
    else:
        return np.array([filter.compute_blocked_distances(i, metric, block,
            X[ref_index], use_max=use_max) for i, block in enumerate(X)])

def get_filter(filter_name, frame_shape, num_blocks):
    if filter_name == 'hist':
        return BlockedHistogram(frame_shape, num_blocks)
    elif filter_name == 'raw':
        return BlockedRawImage(frame_shape, num_blocks)
    else:
        import sys
        print 'Invalid model type: %s' % filter_name
        sys.exit(1)

def train_model(model_name, X_train, Y_train, reg='l2'):
    class_weights = noscope.DataUtils.get_class_weights(Y_train)
    if model_name == 'lr':
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(class_weight=class_weights, penalty=reg)
        model = lr.fit(X_train, Y_train)
    #elif model_name == 'svm':
    #    from sklearn.svm import SVC
    #    svm = SVC(kernel='linear', class_weight=class_weights, probability=True)
    #    model = svm.fit(X_train, Y_train)
    else:
        import sys
        print 'Invalid model type: %s' % model_name
        sys.exit(1)
    return model

def get_confidences(filter, model_name, reg, DELAY, ref_index, metric, X_train_feats,
        X_test_feats, Y, fname_template):
    use_max = model_name == 'none'
    X_all_feats = np.concatenate([X_train_feats, X_test_feats])
    X_distances = get_distances(filter, DELAY, ref_index, metric, X_all_feats,
            use_max)
    train_ind = X_train_feats.shape[0]
    if ref_index == -1:
        # with DELAY = 30, num_frames = 1000, train_ratio = 0.3
        # X_train_distances = dist(30-0),...,dist(300-270)
        # X_test_distances = dist(301-271),...,dist(1000-970)
        # Y_train_distances = Y[30:300]

        X_train_distances, X_test_distances = X_distances[:train_ind-DELAY], \
            X_distances[train_ind-DELAY:]
        Y_train_distances = Y[DELAY:train_ind]
    else:
        X_train_distances, X_test_distances = X_distances[:train_ind], X_distances[train_ind:]
        Y_train_distances = Y[:train_ind]
    if use_max:
        if ref_index == -1:
            confidences = np.concatenate([np.zeros(DELAY), X_distances]).astype('float32')
        else:
            confidences = X_distances

        # Normalize to 0-1
        confidences -= np.min(confidences)
        if np.max(confidences) > 0:
            confidences /= np.max(confidences)
        return confidences
    else:
        model = train_model(model_name, X_train_distances, Y_train_distances,
                reg=reg)
        with open(fname_template + '.model', 'w') as f:
            print >> f, ' '.join(str(val) for val in model.coef_[0])
        probs = model.predict_proba(X_distances)
        if ref_index == -1:
            return np.concatenate([np.zeros(DELAY), probs[:, 1]])
        else:
            return probs[:, 1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_in', required=True, help='CSV with labels')
    parser.add_argument('--video_in', required=True, help='Video input')
    parser.add_argument('--objects', required=True, help='Objects to classify. Comma separated')
    parser.add_argument('--num_frames', type=int, required=True, help='Number of frames')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--base_name', required=True, help='Base output name')
    parser.add_argument('--filters', required=True, help='Filters to try: \
            BlockedHistogram (hist), or BlockedRawImage (raw). Separate with \
            commas.')
    parser.add_argument('--model', required=True, help='Model to try: either \
            Logistic Regression (lr), or take the max of the inputs (none)')
    parser.add_argument('--reg', required=True, help='Regularization param to go \
    with model: either L2 (l2), or L1 (l1). If model is "none", this argument is \
    ignored.')
    parser.add_argument('--num_blocks', type=int, required=True, help='Number of blocks')
    parser.add_argument('--resol', type=int, required=True, help='Resolution. Square')
    parser.add_argument('--delay', type=int, required=True, help='Delay')
    parser.add_argument('--metric', help='Only evaluate one metric')
    parser.add_argument('--ref_index', type=int, required=True, help='Index of reference \
            image to compare against. Must be less than num_frames. If value is \
            not -1, this overrides --delay.')
    args = parser.parse_args()

    DELAY = args.delay
    assert args.ref_index < args.num_frames
    filters = args.filters.split(',')
    objects = args.objects.split(',')
    # for now, we only care about one object, since
    # we're only focusing on the binary task
    assert len(objects) == 1

    print 'Preparing data....'
    data, nb_classes = noscope.DataUtils.get_data(
            args.csv_in, args.video_in,
            binary=True,
            num_frames=args.num_frames,
            OBJECTS=objects,
            regression=True,
            center=False,
            resol=(args.resol, args.resol),
            train_ratio=0.3)
    X_train, Y_train, X_test, Y_test = data
    Y = np.concatenate([Y_train, Y_test])
    Y = Y.flatten()
    Y = Y.astype('uint8')

    mkdir_p(args.output_dir)
    base_fname = os.path.join(args.output_dir, args.base_name)
    print 'Computing features....'
    for filter_name in filters:
        print filter_name
        filter = get_filter(filter_name, X_train[0].shape, args.num_blocks)

        begin = time.time()
        X_train_feats = np.array([filter.compute_feature(X) for X in X_train])
        X_test_feats = np.array([filter.compute_feature(X) for X in X_test])
        end = time.time()
        print end - begin

        for metric_name, metric in filter.distance_metrics():
            if args.metric is not None and metric_name != args.metric: continue
            if metric_name == 'stacked' and args.model == 'none': continue
            if args.ref_index != -1:
                ref_index = find_ref_index(Y_train)
                if ref_index < args.ref_index - 60 or args.ref_index + 60 < ref_index:
                    print 'WARNING',
                print '--ref_index was %d, found %d' % (args.ref_index, ref_index)
            else:
                ref_index = -1

            fname_template = '%s_%s_%s_%s_%s_delay%d_resol%d_ref-index%d_num-blocks%d' % (base_fname,
                    filter_name, metric_name, args.model, args.reg, DELAY,
                    args.resol, ref_index, args.num_blocks)
            csv_fname = fname_template + '.csv'
            print csv_fname
            begin = time.time()
            confidences = get_confidences(filter, args.model,
                    args.reg, DELAY, ref_index, metric, X_train_feats,
                    X_test_feats, Y, fname_template)
            end = time.time()
            print end-begin
            noscope.DataUtils.confidences_to_csv(csv_fname, confidences, objects[0])

if __name__ == '__main__':
    main()
