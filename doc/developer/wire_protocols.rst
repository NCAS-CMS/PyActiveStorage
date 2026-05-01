Wire Protocols
==============

PyActiveStorage has two wire-level protocols in use:

1. **p5rem protocol** for session-based remote file and chunk access over
   stdin/stdout via SSH.
2. **Reductionist protocol** for HTTP-based active chunk reduction.

The data access elements of the protocol are discussed in the first two sections below. 
SSH session setup, remote environment discovery, and control operations are covered in
the third section.

.. toctree::
   :maxdepth: 1

   wire_protocols_p5rem
   wire_protocols_reductionist
  wire_protocols_p5rem_session_setup
