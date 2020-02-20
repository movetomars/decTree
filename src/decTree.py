#!/usr/bin/env python

# YOU DO NOT HAVE TO CHANGE THIS FILE

class DecTree:
    def __init__(self, text, left=None, right=None):
        self.txt = str(text)  ## The name of the attribute or the prediction for that leaf
        self.l = left  ## Left subtree (txt == 0)
        self.r = right  ## Right subtree (txt == 1)

    def __str__(self):
        return self.to_string(0)

    def to_string(self, depth=0):
        if self.is_leaf(): return self.txt

        extra_l = ""
        if not self.l.is_leaf(): extra_l = "\n"
        extra_r = ""
        if not self.r.is_leaf(): extra_r = "\n"

        return depth * "| " + self.txt + " = 0 : " + extra_l + \
               self.l.to_string(depth + 1) + "\n" + \
               depth * "| " + self.txt + " = 1 : " + extra_r + \
               self.r.to_string(depth + 1)

    def is_leaf(self):
        # A tree is a leaf if it doesn't have children
        return self.l is None and self.r is None


if __name__ == "__main__":
    t = DecTree('attr1',
                DecTree('attr2', DecTree('attr3', DecTree('0'), DecTree('1')), DecTree('0')),
                DecTree('attr3', DecTree('0'), DecTree('1')),
                )
    print(t)
