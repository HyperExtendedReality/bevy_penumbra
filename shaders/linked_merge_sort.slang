// Bottom-up stable linked merge sort: O(n log n), no recursion or scratch buffer.
// Each caller exclusively owns every node reachable from its input head.
uint gpuSortLinkedList(uint head) {
    const uint end = 0xffffffffu;
    if (head == end) return head;
    uint width = 1;
    for (uint pass = 0; pass < 32; ++pass) {
        uint input = head, output = end, tail = end, runs = 0;
        while (input != end) {
            ++runs;
            uint left = input, right = input, leftCount = 0;
            for (uint i = 0; i < width && right != end; ++i) {
                ++leftCount; right = gpuSortNext(right);
            }
            uint rightCount = width;
            while (leftCount > 0 || (rightCount > 0 && right != end)) {
                uint selected;
                if (leftCount == 0) {
                    selected = right; right = gpuSortNext(right); --rightCount;
                } else if (rightCount == 0 || right == end || !gpuSortLess(right, left)) {
                    selected = left; left = gpuSortNext(left); --leftCount;
                } else {
                    selected = right; right = gpuSortNext(right); --rightCount;
                }
                if (tail == end) output = selected;
                else gpuSortSetNext(tail, selected);
                tail = selected;
            }
            input = right;
        }
        gpuSortSetNext(tail, end);
        head = output;
        if (runs <= 1) return head;
        width *= 2;
    }
    return head; // WGSL validation requires a return after the unbounded loop.
}

