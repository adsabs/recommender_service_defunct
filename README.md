[![Build Status](https://travis-ci.org/adsabs/recommender_service.svg?branch=master)](https://travis-ci.org/adsabs/recommender_service)

Flask service implementation of recommender service. Start application with

    python wsgi.py

and the do a request like

    curl http://localhost:4000/2010MNRAS.409.1719J

which should return a results like

    {"paper": "2010MNRAS.409.1719J", "recommendations": [{"bibcode": "1998ApJ...509..212S", "author": "Strong,+", "title": "Propagation of Cosmic-Ray Nucleons in the Galaxy"}, {"bibcode": "1998ApJ...493..694M", "author": "Moskalenko,+", "title": "Production and Propagation of Cosmic-Ray Positrons and Electrons"}, {"bibcode": "2007ARNPS..57..285S", "author": "Strong,+", "title": "Cosmic-Ray Propagation and Interactions in the Galaxy"}, {"bibcode": "2011ApJ...737...67M", "author": "Murphy,+", "title": "Calibrating Extinction-free Star Formation Rate Diagnostics with 33 GHz Free-free Emission in NGC 6946"}, {"bibcode": "1971JGR....76.7445R", "author": "Rygg,+", "title": "Balloon measurements of cosmic ray protons and helium over half a solar cycle 1965-1969"}, {"bibcode": "1997ApJ...481..205H", "author": "Hunter,+", "title": "EGRET Observations of the Diffuse Gamma-Ray Emission from the Galactic Plane"}, {"bibcode": "1978MNRAS.182..147B", "author": "Bell,+", "title": "The acceleration of cosmic rays in shock fronts - I."}]}
