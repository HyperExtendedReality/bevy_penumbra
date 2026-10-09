[[vk::binding(0,0)]] StructuredBuffer<uint2> lists;
[[vk::binding(1,0)]] RWStructuredBuffer<uint4> nodes;
[[vk::binding(2,0)]] RWStructuredBuffer<uint> sorted;
uint gpuSortNext(uint index) { return nodes[index].y; }
void gpuSortSetNext(uint index, uint next) { nodes[index].y = next; }
bool gpuSortLess(uint left, uint right) {
    int a = int(nodes[left].x), b = int(nodes[right].x);
    return a < b || (a == b && left < right);
}
#include "linked_merge_sort.slang"
[shader("compute")]
[numthreads(64,1,1)]
void checkLinkedSort(uint3 thread : SV_DispatchThreadID) {
    uint count, stride; lists.GetDimensions(count, stride);
    if (thread.x >= count) return;
    uint2 list = lists[thread.x];
    uint head = gpuSortLinkedList(list.x);
    for (uint i = 0; i < list.y; ++i) {
        sorted[thread.x * 512 + i] = head;
        if (head != 0xffffffffu) head = nodes[head].y;
    }
    sorted[thread.x * 512 + list.y] = head;
}
