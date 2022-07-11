def get_true_label_list(_strict=True):
    # Considering True cases in strict manner
    if _strict:
        true_label_list = ["true", "mostly true",
                   "geppetto checkmark", "one pinocchio"]

    # Considering True cases in Lenient manner
    else:
        true_label_list = ["true", "mostly true",
                   "geppetto checkmark", "one pinocchio",
                   "two pinocchios" , "half true",
                   "half flip"]

    return true_label_list

def get_label_list(_dataset, _ignore, _ignore_under):
    # collect entire labels
    all_labels = {}
    for l in _dataset['train']['review_rating']:
        if l in all_labels:
            all_labels[l] += 1
        else:
            all_labels[l] = 1

    # collect labels except rare cases based on '_ignore_under' parameter
    if _ignore == True:
        ignored_labels = all_labels.copy()
        for l in all_labels:
            if all_labels[l] <= _ignore_under:
                ignored_labels.pop(l)
        return ignored_labels.keys()

    return all_labels.keys()


def revise_dataset(_dataset, _strict=True, _ignore=True, _ignore_under=5):
    # label collection method (strict/lenient)
    true_label_list = get_true_label_list(_strict)

    # a set of considered label list
    label_list = get_label_list(_dataset, _ignore, _ignore_under)

    datasamples = []
    for _idx, _label in enumerate(_dataset['train']['review_rating']):
        if _label in label_list:
            if _label.lower() in true_label_list:
                _label = "True"
            else:
                _label = "Fake"

            _author = _dataset['train'][_idx]['claim_author_name']
            _text = _dataset['train'][_idx]['claim_text']

            datasamples.append({"text": _text,
                             "author": _author,
                             "label": _label})

    return datasamples
