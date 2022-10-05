/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

namespace fvs {

Engine::Engine(Game *game, madrona::WorkerInit &&init)
    : madrona::CustomContext<Engine>(std::move(init)),
      game_(game)
{}

}
