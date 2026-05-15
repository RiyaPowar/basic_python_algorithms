from collections import deque

def shortestPathLength(graph):
    n = len(graph)

    final_mask = (1<<n)-1

    # print(f'final_mask::::::::{final_mask}')

    queue = deque()

    visited = set()

    for i in range(n):
        mask  = (1<<i)
        queue.append((i,mask))
        print(f"queue:::::{queue}")
        visited.add((i,mask))
        print(f"visited:::::{visited}")

    steps = 0

    while queue:
        for _ in range(len(queue)):
            node, mask = queue.popleft()
            print(f"node:::::{node} mask::::{mask}")

            if mask == final_mask:
                return steps
            
            for neighbor in graph[node]:
                print(f"graph:::::::{graph[node]}")
                print(f"neighbor:::::{neighbor}")
                next_mask = mask | (1<<neighbor)
                print(f"next_mask:::::{next_mask}")
                state = (neighbor, next_mask)

                if state not in visited:
                    visited.add(state)
                    queue.append(state)
                    print(f"queue:::::{queue}")
                    print(f"visited:::::{visited}")

        steps += 1
    return -1







graph = [[1,2,3],[0],[0],[0]]
print(shortestPathLength(graph))
