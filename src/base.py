class Base:
    def get_stats(ids, counts=None):
        counts = {} if counts is None else counts
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(ids, sub, id):
        newIds = []
        i = 0
        while i < len(ids):
            if i == len(ids) - 1:
                newIds.append(ids[i])
                break
            elif ids[i] == sub[0] and ids[i + 1] == sub[1]:
                newIds.append(id)
                i += 2
            else:
                newIds.append(ids[i])
                i += 1

        return newIds
