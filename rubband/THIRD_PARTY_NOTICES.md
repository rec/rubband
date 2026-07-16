# Third-Party Notices

Binary wheels for `rubband` may include native libraries copied into the wheel
by the platform wheel repair step.

## Rubber Band Library

`rubband` links to and may distribute the Rubber Band Library, written by
Chris Cannam and published by Particular Programs Ltd.

Rubber Band is distributed under GPL-2.0-or-later. The full GPL 2.0 license
text is included in this package as `LICENSE`.

Source code for Rubber Band is available at:

https://github.com/breakfastquay/rubberband

The release workflow currently builds or installs Rubber Band 4.x for each
platform before building `rubband` wheels. When the wheel repair tools include
Rubber Band or its shared-library dependencies in a wheel, those binaries are
distributed under their original license terms.
