@ti.kernel
def compute_b():
    b = a * 2

@ti.kernel
def compute_c():
    c = b * 2

# forward pass

compute_b()
compute_c()

# backward pass

# this doesn't work
compute_c.grad()
compute_b.grad()

# this works
compute_b.grad()
compute_c.grad()
