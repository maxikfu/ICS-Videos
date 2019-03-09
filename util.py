
def merge(arr, temp, leftStart, rightEnd):
    leftEnd = (rightEnd + leftStart) // 2
    rightStart = leftEnd + 1
    inversions = 0
    left = leftStart
    right = rightStart
    index = leftStart
    size = rightEnd - leftStart + 1
    while left <= leftEnd and right <= rightEnd:
        if arr[left] <= arr[right]:
            temp[index] = arr[left]
            left += 1
        else:
            temp[index] = arr[right]
            inversions += leftEnd - left + 1
            right += 1
        index += 1
    for x in arr[left:leftEnd+1]:
        temp[index] = x
        index += 1
    for x in arr[right:rightEnd+1]:
        temp[index] = x
        index += 1
    arr[leftStart:rightEnd+1] = temp[leftStart:rightEnd+1]
    return inversions



def mergeSort(arr, temp, leftStart, rightEnd):
    inversions = 0
    if leftStart >= rightEnd:
        return inversions
    split = (leftStart + rightEnd) // 2
    inversions += mergeSort(arr, temp, leftStart, split)
    inversions += mergeSort(arr, temp, split + 1, rightEnd)
    inversions += merge(arr, temp, leftStart, rightEnd)
    return inversions


# Complete the countInversions function below.
def countInversions(arr):
    temp = [None]*len(arr)
    split = (len(arr)-1) // 2
    leftStart = 0
    rightEnd = len(arr) - 1
    ans = 0
    ans += mergeSort(arr, temp, leftStart, split)
    ans += mergeSort(arr, temp, split + 1, rightEnd)
    ans += merge(arr, temp, leftStart, rightEnd)

    print(ans)


if __name__ == '__main__':
    ar = [5, 4, 3, 2, 1]
    countInversions(ar)
