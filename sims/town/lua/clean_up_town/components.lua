--[[ Components for the clean_up_town level (new-melting-pot, experiment 1).

Everything from Clean Up, plus:
  * CleanerBadge: the "connecting AI". A small badge rides on a player's avatar
    for a while after they clean the river, so everyone can see who's pitching in.
  * Feed: the "isolating AI". A tile that pays a small private reward every few
    frames to whoever stands on it, like an engagement feed.

Builds on Melting Pot's own pieces: the badge follows its avatar the same way
Allelopathic Harvest's marking overlays do (AvatarConnector), and it reads Clean
Up's Cleaner.player_cleaned, which DirtCleaning:onHit sets when a beam hits dirt.
]]

local args = require 'common.args'
local class = require 'common.class'
local events = require 'system.events'

local meltingpot = 'meltingpot.lua.modules.'
local component = require(meltingpot .. 'component')
local component_registry = require(meltingpot .. 'component_registry')

-- Register all of Clean Up's components first.
local clean_up = require 'meltingpot.lua.levels.clean_up.components'


--[[ CleanerBadge lives on a badge object connected to one avatar.

Arguments:
`playerIndex` (int): the avatar this badge belongs to (Lua, 1-based).
`onState` (str): badge state that shows the badge.
`offState` (str): badge state that hides it (same layer, transparent sprite).
`waitState` (str): badge state while the avatar is zapped out (no layer).
`framesVisible` (int): how long the badge stays up after the last cleaning.
`layer` (str): render layer for the badge; drawn after (on top of) avatars.
]]
local CleanerBadge = class.Class(component.Component)

function CleanerBadge:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('CleanerBadge')},
      {'playerIndex', args.numberType},
      {'onState', args.stringType},
      {'offState', args.stringType},
      {'waitState', args.stringType},
      {'framesVisible', args.positive},
      {'layer', args.default('badgeLayer'), args.stringType},
  })
  CleanerBadge.Base.__init__(self, kwargs)
  self._config.playerIndex = kwargs.playerIndex
  self._config.onState = kwargs.onState
  self._config.offState = kwargs.offState
  self._config.waitState = kwargs.waitState
  self._config.framesVisible = kwargs.framesVisible
  self._config.layer = kwargs.layer
end

function CleanerBadge:addHits(worldConfig)
  -- Appending the layer puts it last in the render order, above the avatars
  -- (Clean Up's avatars are on `superOverlay`).
  component.insertIfNotPresent(worldConfig.renderOrder, self._config.layer)
end

function CleanerBadge:reset()
  self._timer = 0
end

function CleanerBadge:registerUpdaters(updaterRegistry)
  local function update()
    local sim = self.gameObject.simulation
    local avatarObject = sim:getAvatarFromIndex(self._config.playerIndex)
    local cleaner = avatarObject:getComponent('Cleaner')
    if (cleaner.player_cleaned or 0) > 0 then
      self._timer = self._config.framesVisible
    elseif self._timer > 0 then
      self._timer = self._timer - 1
    end
    local state = self.gameObject:getState()
    if state == self._config.waitState then
      return  -- Avatar is zapped out; AvatarConnector restores us on respawn.
    end
    if self._timer > 0 and state ~= self._config.onState then
      self.gameObject:setState(self._config.onState)
    elseif self._timer == 0 and state ~= self._config.offState then
      self.gameObject:setState(self._config.offState)
    end
  end
  -- Higher priority runs earlier in a frame (updater_registry.lua). A beam's
  -- hit (DirtCleaning:onHit -> Cleaner:setCumulant) lands after the updaters,
  -- and Cleaner resets the cumulant at priority 400 at the start of the next
  -- frame. So we read it at 450, before the reset: the badge shows one frame
  -- after the cleaning hit.
  updaterRegistry:registerUpdater{updateFn = update, priority = 450}
end

function CleanerBadge:isShowing()
  return self.gameObject:getState() == self._config.onState
end


--[[ Feed pays whoever stands on it `rewardAmount` every `period` frames.

Arguments:
`rewardAmount` (number): reward per payout.
`period` (int): frames between payouts while someone stays on the tile.
]]
local Feed = class.Class(component.Component)

function Feed:__init__(kwargs)
  kwargs = args.parse(kwargs, {
      {'name', args.default('Feed')},
      {'rewardAmount', args.numberType},
      {'period', args.positive},
  })
  Feed.Base.__init__(self, kwargs)
  self._config.rewardAmount = kwargs.rewardAmount
  self._config.period = kwargs.period
end

function Feed:reset()
  self._occupant = nil
  self._timer = 0
end

function Feed:onEnter(enteringGameObject, contactName)
  if contactName == 'avatar' then
    self._occupant = enteringGameObject
    self._timer = self._config.period
  end
end

function Feed:onExit(exitingGameObject, contactName)
  if contactName == 'avatar' and exitingGameObject == self._occupant then
    self._occupant = nil
  end
end

function Feed:registerUpdaters(updaterRegistry)
  local function update()
    local occupant = self._occupant
    if occupant == nil then
      return
    end
    local avatar = occupant:getComponent('Avatar')
    if not avatar:isAlive() then
      self._occupant = nil
      return
    end
    local here = self.gameObject:getPosition()
    local there = occupant:getPosition()
    if here[1] ~= there[1] or here[2] ~= there[2] then
      self._occupant = nil
      return
    end
    self._timer = self._timer - 1
    if self._timer <= 0 then
      avatar:addReward(self._config.rewardAmount)
      events:add('feed_paid', 'dict', 'player_index', avatar:getIndex())
      self._timer = self._config.period
    end
  end
  updaterRegistry:registerUpdater{updateFn = update, priority = 150}
end


local allComponents = {
    CleanerBadge = CleanerBadge,
    Feed = Feed,
}
component_registry.registerAllComponents(allComponents)

-- Return Clean Up's components too, so the level sees one table.
for name, value in pairs(clean_up) do
  allComponents[name] = value
end
return allComponents
