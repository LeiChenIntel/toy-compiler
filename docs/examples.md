# Examples

##### 1. Add values

This is the example to add two integers and return the result. C and C++ API are included independently.

It can be run directly by

```bash
add-values.exe
```

Then logs are dumped as

```text
TEST: Add Value C API
Input: 41 Result: 82
TEST: Add Value CPP API
Input: 42 Result: 84
```

##### 2. Add tensors

This is an example to add two tensors with 3 elements. Both of the tensors are imported as pointers. At last the result
values are stored to the buffer which is imported as an input pointer. The execution file needs a Toy language
description of these operations, and an example is shown in `example.toy`.

An example of `example.toy` can be given by

```text
def funcInputs() {
  var a<3> = [0];
  var b<3> = [0];
  var c<3> = [0];
}

def add_tensors(a, b, c) {
  var d = a + b;
  store(d, c);
  return;
}
```

It can be run by

```bash
add-tensors.exe example.toy
```

The output result is
```text
3.000000 5.000000 7.000000
```

Note that although the example has a name `add-tensors`, it is more general as it named.
If the addition operation is changed to multiplication operation. It works.
With an operation,
```text
var d = a .* b;
```

The result can be
```text
0.000000 4.000000 10.000000
```
