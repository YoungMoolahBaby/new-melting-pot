--[[ clean_up_town: Clean Up plus a connecting AI (badges) and an isolating AI
(feed tiles). New-melting-pot, experiment 1. Same shape as Melting Pot's
clean_up/init.lua (Apache 2.0).
]]

local meltingpot = 'meltingpot.lua.modules.'
local api_factory = require(meltingpot .. 'api_factory')
local simulation = require(meltingpot .. 'base_simulation')

-- Required so their components are registered.
local component_library = require(meltingpot .. 'component_library')
local avatar_library = require(meltingpot .. 'avatar_library')
local components = require 'components'

return api_factory.apiFactory{
    Simulation = simulation.BaseSimulation,
    settings = {
        spriteSize = 8,
        maxEpisodeLengthFrames = 1000,
        simulation = {},
    }
}
